import { useState } from 'react';
import {
  StyleSheet,
  Text,
  View,
  SafeAreaView,
  Image,
  TouchableWithoutFeedback,
  TouchableOpacity,
  Button,
  Alert, // an API, not component
  Platform,
  StatusBar,
} from 'react-native';

export default function App() {
  console.log('hello');
  const [count, setCount] = useState(0);
  const [name, setName] = useState('Dan');

  const incrementCount = () => {
    setCount((prev) => {
      prev + 1 === 10 && alert('you counted to 10!');
      // custom alert:
      prev + 1 === 15 &&
        Alert.alert('Yay', 'you counted to 15!', [
          { text: 'start over', onPress: (value) => setCount(0) },
          { text: 'continue' },
        ]);

      // note: Alert.prompt() only works on IOS!
      // prev + 1 === 20 &&
      //   Alert.prompt('High Score!', "What's your name?", (text) =>
      //     setName(text)
      //   );
      return prev + 1;
    });
  };

  return (
    // https://reactnative.dev/docs/components-and-apis
    // SafeAreaView ensures content is not hidden by a notch in the phone screen (ONLY iPhone)
    <>
      <SafeAreaView style={styles.safeHeader}>
        <Text style={{}} onPress={() => console.log('clicked text!')}>
          Hello {name}! {count > 0 ? ` X ${count}` : ''}
        </Text>
      </SafeAreaView>

      <SafeAreaView style={styles.container}>
        {/* local image */}

        {/* briefly lowers opacity of content when pressed */}
        <TouchableWithoutFeedback>
          <Image
            style={{ width: 40, height: 40 }}
            source={require('./assets/icon.png')}
          />
        </TouchableWithoutFeedback>

        {/* network image */}
        <TouchableOpacity onPress={incrementCount}>
          <Image
            source={{
              uri: 'https://picsum.photos/200/300',
              width: 200,
              height: 300,
            }}
          />
        </TouchableOpacity>

        {count ? (
          <Button title="Reset Count" onPress={() => setCount(0)} />
        ) : null}
      </SafeAreaView>
    </>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    // marginTop: 40,
    backgroundColor: 'dodgerblue',
    alignItems: 'center',
    justifyContent: 'center',
    // on IOS, the SafeAreView automatically handles this. with android we need to do more work
  },

  safeHeader: {
    paddingTop: Platform.OS === 'android' ? StatusBar.currentHeight : 0,
    backgroundColor: 'darkorange',
    height: 65,
    alignItems: 'center',
    justifyContent: 'center',
  },
});
