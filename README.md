# Docker 

## Build and run locally

### Build

```bash
docker build . -f ./Dockerfile -t my-image:my-tag
```

### Run

```bash
docker run -v /mnt/e/AI_Prcp_test1/:/data -t my-image:my-tag
``` 
