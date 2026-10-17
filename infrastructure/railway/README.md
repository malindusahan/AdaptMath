# Railway deployment

Service-specific Railway configuration will be added here after the component
restructure is stable. The planned services are Web, Tutor API, Student Memory
API, and PostgreSQL.

The Tutor build must use the repository root as its Docker build context because
it consumes both directories under `packages/`.
