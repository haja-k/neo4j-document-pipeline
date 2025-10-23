# Neo4j Document Injection

## Getting started

### ➡️ In development environment (local development):
1. Build the docker image by running the file **build_images.sh**
2. Run the file named **run_dev.sh** 
3. Observe the logs from the terminal and fix/update the API accordingly
4. Access the Swagger UI to see the API list at **http://127.0.0.1:8086/docs**

### ➡️ In TNT environment (Server deployment):
1. Build the docker image by running the file **build_images.sh**
2. Run the command ```docker compose -f docker-compose.dev.yml up -d --build```
3. Observe the logs from the terminal by running the command ```docker compose -f logs```

### ➡️ In production environment:
1. Build the docker image by running the file **build_images.sh**
2. Run the file named **run_prod.sh** 

## Note:
- Running sh files in Linux and Windows have different methods respectively.
- In Linux OS, make the file executable first by ```chmod +x <file name>```
- In Windows terminal, just type in the name of the file