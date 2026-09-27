#include <pebble.h>


// "s_" hints "static" (file-scoped)
static window *s_main_window;

static void main_window_load(Window *window) {

}

static void main_window_unload(Window *window) {

}


static void init() {
  s_main_window = window_create();
  window_set_background_color(s_main_window, GColorBlue);

  // handlers for managing window
  //window_set_window_handlers(s_main_window
}

static void deinit() {
  window_destroy(s_main_window)`
}


int main(void) {
  init();

  app_event_loop();
  deinit();
}
