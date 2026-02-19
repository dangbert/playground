Simple demo on a single-node kubernetes cluster running mongodb and a demo webapp.

**NOT FULLY TESTED** next step would be to run minikube with qemu emulating amd64

references:
* [video tutorial](https://youtu.be/s_o8dwzRlu4?t=2477), and accompanying [repo](https://gitlab.com/nanuchi/k8s-in-1-hour)


## How to run

````bash
minikube start --driver docker

kubectl apply -f mongo-config.yaml
kubectl apply -f mongo-secret.yaml

kubectl apply -f mongo.yaml
kubectl apply -f webapp.yaml

# check status
kubectl get all
kubectl get configmap && kubectl get secret

# check logs
kubectl logs -f webapp-deployment-changeme

# get IP of cluster node (minikube)
minkube ip
# OR check "INTERNAL-IP" here:
kubectl get node -o wide
````

And visit the IP on port 33000 e.g. http://192.168.12.34:33000

