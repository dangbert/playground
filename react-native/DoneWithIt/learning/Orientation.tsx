import { useState } from 'react';
import {
  Dimensions,
  StyleSheet,
  SafeAreaView,
  Text,
  Platform,
  StatusBar,
} from 'react-native';

import {
  useDeviceOrientation,
  useDimensions,
} from '@react-native-community/hooks';

/**
 * Experiments with detecting phone orientation and screen size.
 * Note: "orientation" must be set to "defualt" in app.json (to allow portrait and landscape).
 */
export default function Orientation() {
  // dimensions of screen (rotation independent)
  console.log(Dimensions.get('window'));

  // dimensions of screen that actually consider rotation:
  const dimensions = useDimensions();
  console.log(dimensions);

  // const isPortrait = dimensions.window.height > dimensions.window.width;
  const isPortrait: boolean = useDeviceOrientation().portrait; // quicker method

  return (
    <SafeAreaView style={styles.container}>
      {/* <View></View> */}
      <Text>hello world ({isPortrait ? 'portrait' : 'landscape'})!</Text>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    // marginTop: 40,
    backgroundColor: 'dodgerblue',
    alignItems: 'center',
    justifyContent: 'center',

    paddingTop: Platform.OS === 'android' ? StatusBar.currentHeight : 0,
  },
});
