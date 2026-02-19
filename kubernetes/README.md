
[install minikube](https://minikube.sigs.k8s.io/docs/start/?arch=%2Flinux%2Fx86-64%2Fstable%2Fbinary%20download)

````bash
# comes with kubectl
brew install minikube

# for qemu network
brew install socket_vmnet

# https://github.com/lima-vm/socket_vmnet/issues/18#issuecomment-1574149506
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --unblock /usr/libexec/bootpd


#/opt/homebrew/bin/socket_vmnet
````

````bash
# start (local) cluster
minikube start --driver docker


# on MAC you could also try:
minikube start --driver=qemu --network=socket_vmnet


# get cluster status
kubectl get pods -A
````

### what didn't work
`socket_vmnet` installed with brew but couldn't be found by minikube.

````bash

````
