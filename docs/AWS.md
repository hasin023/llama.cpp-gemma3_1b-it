# Load Testing with Locust on AWS

This guide outlines the steps to deploy the LLM service on an AWS EC2 instance and run load tests using Locust.

## 1. Setup AWS Instance

- **Name**: Example `llama-service-load-test`
- **Region**: Select `us-east-1`
- **Operation System**: Select `Ubuntu Server 24.04 LTS`
- **Instance Type**: Select `c8g.xlarge` (graviton4 series) or `t4g.xlarge` (graviton2 series).
- **Storage**: Set storage to `30GB - gp3`.
- **Key Pair**: Create a new key pair:
  - Type: `RSA`
  - Format: `pem`
  - This will download a `.pem` file. Keep it safe.
- **Security Group**:
  - Allow **SSH (Port 22)** from `0.0.0.0/0`.
  - Allow **HTTP (Port 80)** from `0.0.0.0/0` (for application access).

## 2. Connect to the Instance

### Linux / Mac

First, set permissions for your key file:

```bash
sudo chmod 400 ~/your-key-pair.pem
```

Connect using SSH:

```bash
ssh -i ~/your-key-pair.pem ubuntu@<your-instance-public-ip>
```

### Windows (PowerShell)

Set permissions for the key file:

```powershell
icacls "$env:USERPROFILE\your-key-pair.pem" /inheritance:r
icacls "$env:USERPROFILE\your-key-pair.pem" /grant:r "$($env:USERNAME):R"
```

Connect using SSH:

```powershell
ssh -i "$env:USERPROFILE\your-key-pair.pem" ubuntu@<your-instance-public-ip>
```

> [!TIP]
> For Windows users, the preferred approach is to use **WSL (Windows Subsystem for Linux)**, as it allows you to follow the Linux instructions directly.

## 3. Environment Setup

### Git Configuration (Private Repo Access)

Inside the instance, generate an SSH key:

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

Add the private key to the agent:

```bash
ssh-add ~/.ssh/id_ed25519
```

View the public key:

```bash
cat ~/.ssh/id_ed25519.pub
```

Copy the output and add it to your GitHub repository's **Deploy keys** (Read-only access).

Test the connection:

```bash
ssh -T git@github.com
```

### Install Git (if missing)

```bash
sudo apt install git-all
```

Verify installation:

```bash
git --version
```

### Install Docker & Docker Compose

Add Docker's official repository:

```bash
# Add Docker's official GPG key:
sudo apt update
sudo apt install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt update
```

Install packages:

```bash
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Verify Docker status:

```bash
sudo systemctl status docker
# If not running, start it using the following command:
# sudo systemctl start docker
```

Verify versions:

```bash
docker --version
docker compose version
```

## 4. Application Deployment

Clone the repository:

```bash
mkdir workspace
cd workspace
git clone <your-repository-url>
```

Enter the directory:

```bash
cd <repo-directory>
```

_(Note: Replace `<repo-directory>` with your actual folder name)_

Setup environment:

```bash
cp .env.example .env
```

Start the application:

```bash
docker compose up --build -d
```

Verify containers are running:

```bash
docker ps
```

Check logs (if needed):

```bash
docker logs <container_id>
```

## 5. Load Testing

### Run Locust (Headless Mode)

Run the load test from your **local machine** targeting the AWS instance:

```bash
python -m locust -f  .\scripts\locust_llm-service-api.py --headless -u 1 -r 1 -t 3600s --host http://<your-instance-public-ip> --system-config "AWS c8g.xlarge"
```

### Run Locust (GUI Mode)

To run with GUI:

```bash
python -m locust -f  .\scripts\locust_llm-service-api.py
```

- **Local Machine**: Access at `http://localhost:8089`
- **From Instance**: Access at `http://<your-instance-public-ip>:8089`

> [!WARNING]
> Make sure to open **port 8089** in the instance's Security Group if accessing the GUI externally.

### Stop Application

When testing is complete:

```bash
docker compose down
```

## 6. Cleanup

To avoid unnecessary costs:

1. Stop the instance from the AWS Console.
2. Terminate the instance.
3. Remove the Key Pair.
4. Delete the Security Group (do **NOT** delete the `default` group).
