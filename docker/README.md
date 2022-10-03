# Docker GUI Demo

Goal: view a GUI running in docker.

References:
* [video tutorial](https://www.youtube.com/watch?v=rCdd6u_FXnQ)

## Usage:
````bash
./run.sh
````

## Troubleshooting

"Authorization required, but no authorization protocol specified": this is likely an issue with Docker running as root, and Xhost not being allowed to run with root.
* [fix / explanation](https://stackoverflow.com/a/49717572/5500073), [more info](https://www.reddit.com/r/linux4noobs/comments/lu1plx/hi_i_get_this_authorization_required_but_no/?utm_source=share&utm_medium=web2x&context=3)
  * Quick fix is to run `xhost si:localuser:root`, and later revert permissions with `xhost -si:localuser:root`
* (or install docker so it can be ran without `sudo`)

