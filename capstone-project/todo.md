## ArgoCD Docs
https://www.digitalocean.com/community/tutorials/how-to-deploy-to-kubernetes-using-argo-cd-and-gitops     ## Installations

https://argo-cd.readthedocs.io/en/stable/getting_started/       ## Creating Apps through UI

Commands
```bash
kubectl create namespace argocd

kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

watch kubectl get pods -n argocd

kubectl port-forward svc/argocd-server -n argocd 8080:443                    ## Port forwarding to access ArgoCD UI

kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d; echo            ## Retrieve ArgoCD password. User is `admin`
```

### Run SonarQube
```bash
sudo apt install docker.io

sudo usermod -aG docker $USER

newgrp docker 

docker run --name sonarqube -p 9000:9000 -d sonarqube:10.6-community
```

### Amazon EKS