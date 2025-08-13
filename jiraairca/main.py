from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
import requests
import os
import logging
import uuid
import urllib3
import json
from atlassian import Jira
# from atlassian import Confluence
# from atlassian import Crowd
# from atlassian import Bitbucket
# from atlassian import ServiceDesk
# from atlassian import Xray

urllib3.disable_warnings()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load environment variables
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
OLLAMA_API = os.getenv("OLLAMA_API", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "mistral")

JIRA_URL = os.getenv("JIRA_URL")
JIRA_USER = os.getenv("JIRA_USER")
JIRA_TOKEN = os.getenv("JIRA_TOKEN")
JIRA_PAT = os.getenv("JIRA_PAT")

# Init services
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
model = SentenceTransformer("all-MiniLM-L6-v2")
jira_connection = Jira(
    url=JIRA_URL,
    token=JIRA_PAT,
    verify_ssl=False
)

# Vector DB Collection
COLLECTION_NAME = "jira_issues"
VECTOR_SIZE = 384

# Request models
class HydrateRequest(BaseModel):
    issue_keys: List[str]

class IssueRequest(BaseModel):
    issue_key: str
    comment_on_key: str

@app.on_event("startup")
def setup_qdrant_collection():
    collections = qdrant.get_collections().collections
    if COLLECTION_NAME not in [c.name for c in collections]:
        qdrant.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )
        logger.info(f"Qdrant collection '{COLLECTION_NAME}' created.")

def extract_issue_payload(issue):
    fields = issue.get("fields", {})

    return {
        "issue_key": issue.get("key"),
        "title": fields.get("summary"),
        "summary": fields.get("summary"),
        "description": fields.get("description"),

        "comments": "\n".join([
            c.get("body", "") for c in jira_connection.issue_get_comments(issue["key"]).get("comments", [])
        ]),

        # Metadata
        "created": fields.get("created"),
        "updated": fields.get("updated"),
        "creator": fields.get("creator", {}).get("displayName"),
        "reporter": fields.get("reporter", {}).get("displayName"),
        "assignee": fields.get("assignee", {}).get("displayName"),
        "status": fields.get("status", {}).get("name"),
        "project": fields.get("project", {}).get("key"),
        "issue_type": fields.get("issuetype", {}).get("name"),
        "labels": fields.get("labels", []),

        # Context
        "components": [comp.get("name") for comp in fields.get("components", [])],
        "found_in_version": [v.get("name") for v in fields.get("versions", [])],
        "fix_versions": [v.get("name") for v in fields.get("fixVersions", [])],
        "resolution": fields.get("resolution", {}).get("name") if fields.get("resolution") else None,

        # Linked issues
        "linked_issues": [
            link.get("outwardIssue", {}).get("key")
            for link in fields.get("issuelinks", [])
            if link.get("outwardIssue")
        ]
    }

@app.post("/hydrate/")
def hydrate_issues(req: HydrateRequest):
    points = []

    for key in req.issue_keys:
        try:
            issue = jira_connection.issue(key)
            payload = extract_issue_payload(issue)

            text_blob = " ".join([
                payload.get("summary", ""),
                payload.get("description", ""),
                payload.get("comments", ""),
                payload.get("rca", "")
            ])
            embedding = model.encode(text_blob).tolist()

            points.append(PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, key)),
                vector=embedding,
                payload=payload
            ))

        except Exception as e:
            logger.error(f"Failed to process issue {key}: {e}")

    if points:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        return {"status": "success", "ingested": len(points)}
    else:
        return {"status": "failed", "error": "No issues ingested"}

@app.post("/analyze/")
def analyze_jira_issue(req: IssueRequest):
    try:
        issue = jira_connection.issue(req.issue_key)
        payload = extract_issue_payload(issue)

        query_text = " ".join([
            payload.get("summary", ""),
            payload.get("description", ""),
            payload.get("comments", "")
        ])
        query_vector = model.encode(query_text).tolist()

        results = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=3
        )
        similar_issues = [r.payload for r in results]

        # Build prompt
        prompt = f"""You're a senior engineer doing RCA (Root Cause Analysis).

New JIRA Issue:
Title: {payload['title']}
Summary: {payload['summary']}
Description: {payload['description']}

Historical Issues:\n"""

        for i, sim in enumerate(similar_issues):
            prompt += (
                f"\nIssue #{i+1} - {sim.get('issue_key')}\n"
                f"Summary: {sim.get('summary')}\n"
                f"Description: {sim.get('description')}\n"
                f"Comments: {sim.get('comments')}\n"
            )

        prompt += "\nBased on the similarities above, what could be the root cause?"

        # Call LLM
        response = requests.post(
            f"{OLLAMA_API}/api/generate",
            json={"model": MODEL_NAME, "prompt": prompt}
        )
        response.raise_for_status()
        combined_response = ""
        for line in response.text.strip().split('\n'):
            try:
                chunk = json.loads(line)
                combined_response += chunk.get("response", "")
            except json.JSONDecodeError:
                # Handle or log malformed line
                continue

        answer = combined_response.strip()

        jira_connection.issue_add_comment(req.comment_on_key, answer)

        return {
            "issue_key": req.issue_key,
            "root_cause_analysis": answer,
            "similar_issues": similar_issues
        }

    except Exception as e:
        logger.exception("Analysis failed")
        return {"error": str(e)}
