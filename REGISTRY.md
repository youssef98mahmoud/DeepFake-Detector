# How to use Container Registries

## Publishing images

There are two ways to publish an image to a registry, manually, or automatically every time a new version of the project is tagged. The recommended way is automatically via CI/CD.

For either option, you need to set up an access token with write permissions to the repository.

### Manually

Log into the registry using the created token:

```console
$ docker login <REGISTRY_URL>
Username: <USERNAME>
PASSWORD: <TOKEN>
$ docker tag deepfake <REGISTRY_URL>:<TAG>
$ docker push <REGISTRY_URL>:<TAG>
```

### Automatically

#### GitLab Container Registry

Create variables:

- CI_REGISTRY_PASSWORD (masked and hidden) = GitLab access token with read_registry, write_registry permissions

Define pipeline:

```yaml
stages:
  - lint
  - build
  - scan

dockerfile-lint:
  stage: lint
  image: hadolint/hadolint:v2.12.0-debian
  script:
    - hadolint --failure-threshold error Dockerfile
  only:
    - tags
    - branches

build-and-push:
  stage: build
  image:
    name: gcr.io/kaniko-project/executor:v1.22.0-debug
    entrypoint: [""]
  script:
    - echo "{\"auths\":{\"$CI_REGISTRY\":{\"username\":\"$CI_REGISTRY_USER\",\"password\":\"$CI_REGISTRY_PASSWORD\"}}}" > /kaniko/.docker/config.json
    - /kaniko/executor
        --context $CI_PROJECT_DIR
        --dockerfile $CI_PROJECT_DIR/Dockerfile
        --destination $CI_REGISTRY_IMAGE:$CI_COMMIT_TAG
  only:
    - tags

trivy-sbom:
  stage: scan
  image:
    name: aquasec/trivy:0.49.1
    entrypoint: [""]
  before_script:
    - mkdir -p ~/.docker
    - echo "{\"auths\":{\"$CI_REGISTRY\":{\"username\":\"$CI_REGISTRY_USER\",\"password\":\"$CI_REGISTRY_PASSWORD\"}}}" > ~/.docker/config.json
  script:
    - trivy image --scanners vuln --format json --output trivy.sbom.json $CI_REGISTRY_IMAGE:$CI_COMMIT_TAG
  artifacts:
    paths:
      - trivy.sbom.json
  only:
    - tags
```

>**Note:** The pipeline includes tasks for automatically checking if the Dockerfile is valid and scanning the image for any vulnerabilities. This is important for running software on shared services.

#### EIDF Harbor Container Registry

Create variables:

- HARBOR_REGISTRY = registry.eidf.ac.uk
- HARBOR_PROJECT = name of the project on Harbor
- HARBOR_USER = Harbor username
- HARBOR_PASSWORD (masked and hidden) = Harbor CLI secret

Define pipeline:

```yaml
stages:
  - lint
  - build

dockerfile-lint:
  stage: lint
  image: hadolint/hadolint:v2.12.0-debian
  script:
    - hadolint --failure-threshold error Dockerfile
  only:
    - tags
    - branches

build-and-push:
  stage: build
  image:
    name: gcr.io/kaniko-project/executor:v1.22.0-debug
    entrypoint: [""]
  script:
    - echo "{\"auths\":{\"$HARBOR_REGISTRY\":{\"username\":\"$HARBOR_USER\",\"password\":\"$HARBOR_PASSWORD\"}}}" > /kaniko/.docker/config.json
    - /kaniko/executor
        --context $CI_PROJECT_DIR
        --dockerfile $CI_PROJECT_DIR/Dockerfile
        --destination $HARBOR_REGISTRY/$HARBOR_PROJECT/$CI_REGISTRY_IMAGE:$CI_COMMIT_TAG
  only:
    - tags
```

>**Note:** Harbor has integrated Trivy scanning so this doesn't have to be added to the pipeline.

#### Validating scans

You must check the vulnerability report of your image and try to solve any solvable issues. You should aim to solve any high vulnerabilities, however, it is known that this may not be possible with certain base images.
