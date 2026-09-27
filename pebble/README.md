
* [install SDK](https://developer.repebble.com/sdk/)


````bash
# pebble CLI
uv tool install pebble-tool

pebble sdk list
pebble sdk install latest

# install emulator
# see list at https://developer.repebble.com/tutorials/watchface-tutorial/part1/
pebble install --emulator emery

# init project
pebble new-project my-app
cd my-app

pebble build
pebble install ./build/first-app.pbw --emulator emery
````

references:
* https://developer.repebble.com/tutorials/watchface-tutorial/part1/
