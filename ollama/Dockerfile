FROM ollama/ollama

WORKDIR /home

COPY privatedomain_trusted_certs.crt ./

RUN apt-get update && apt-get install -y ca-certificates openssl curl \
  && openssl s_client -showcerts -connect registry.ollama.ai:443 </dev/null 2>/dev/null \
  | openssl x509 -outform PEM > /usr/local/share/ca-certificates/registry.ollama.ai.crt \
  && cp /home/privatedomain_trusted_certs.crt /usr/local/share/ca-certificates/privatedomain_trusted_certs.crt \
  && update-ca-certificates
