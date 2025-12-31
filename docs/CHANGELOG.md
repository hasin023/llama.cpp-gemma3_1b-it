# Changelog

## 2025-12-31 - Production Readiness & AWS Deployment

### Added

- **Nginx Reverse Proxy**:
  - Implemented `nginx` service in `compose.yaml` to handle incoming traffic on port 80.
  - Added `nginx.conf` with configured upstream to `llm-service-api:8000`.
  - Configured proxy timeouts to 600s to handle long-running LLM inference requests.

### Changed

- **Network Security**:
  - Removed public exposure of `llama-cpp-server` (8080) and `llm-service-api` (8000).
  - Services now communicate strictly over the internal Docker network.
  - Only port 80 (Nginx) is exposed to the host.
- **Load Testing**:
  - Updated `scripts/locust_llm-service-api.py` to use wall-clock duration for accurate hardware cost analysis.
  - Updated default host in Locust script to `http://localhost` (port 80).
- **Documentation**:
  - Detailed AWS EC2 deployment guide in [`docs/AWS.md`](AWS.md).
  - Updated `README.md` and `docs/README.md` with new documentation links.
  - Updated `llm_service` configuration to use internal DNS names.

## 2025-12-29 - Codebase Organization & Documentation Cleanup

### Fixed

- **Conversation Display**: Updated `docker_llm-api-service_inference.py` to clearly show user queries and model responses in conversation flow
  - Added formatted conversation display with role labels (👤 User / 🤖 Model)
  - Shows user queries being sent before each API call
  - Displays full conversation history after each turn

### Organized

- **Documentation Structure**:

  - Created `docs/` directory for additional documentation
  - Moved `LlaMacppServer.md` to `docs/` (auto-generated reference)
  - Created `docs/README.md` as documentation index
  - Updated all documentation references to reflect new structure

- **Scripts Organization**:
  - Created `scripts/` directory for test scripts
  - Moved test scripts to `scripts/`:
    - `docker_llm-api-service_inference.py` (LLM Service API tests)
    - `llama-server_docker_inf.py` (Direct llama.cpp completions API tests)
    - `llama-server_docker_inf_chat.py` (Direct llama.cpp chat API tests)

### Removed

- **Unnecessary Files**:
  - Removed root `main.py` (unused hello world script)

### Updated

- **README.md**: Added project structure section and updated test script references
- **LLM-SERVICE.md**: Updated file path references to new structure

### Preserved

- `LOCUST_llama-server_docker_inf.py` remains in root (load testing script)
- `Dockerfile.server` kept for reference (currently commented out in compose.yaml)
- All core functionality preserved
