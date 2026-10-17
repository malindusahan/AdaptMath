# Infrastructure

Deployment definitions live here. `railway/` is reserved for service-specific
Railway configuration; `docker/` is reserved for local container orchestration.
Production services must use Railway variables and private networking rather
than local `.env` files or hard-coded ports.
