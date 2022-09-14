import React from 'react';
import SafestAreaView from '../lib/SafestAreaView';
import {
  StyleSheet,
  Image,
  Text,
  TouchableOpacity,
  View,
  TouchableHighlight,
} from 'react-native';
import { COLORS } from '../constants';

interface WelcomeScreenProps {
  changeScreen: (next: string) => void;
}

const WelcomeScreen: React.FC<WelcomeScreenProps> = ({ changeScreen }) => {
  return (
    <SafestAreaView style={styles.container}>
      <Image
        style={{ ...styles.main, width: '100%', height: '100%' }}
        source={require('../assets/background.jpg')}
      />

      <View style={{ flex: 1, justifyContent: 'flex-end' }}>
        <View
          style={{
            position: 'absolute',
            top: 100,
            alignSelf: 'center',
            alignItems: 'center',
          }}
        >
          <Image
            style={{ width: 120, height: 120, marginBottom: 10 }}
            source={require('../assets/logo-red.png')}
          />
          <Text style={{ fontSize: 20 }}>Sell What You Don't Need</Text>
        </View>

        <View>
          <TouchableHighlight
            style={styles.login}
            onPress={() => changeScreen('image')}
            activeOpacity={0.5}
          >
            <></>
          </TouchableHighlight>
          <TouchableOpacity style={styles.signup} />
        </View>
      </View>
    </SafestAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  main: {
    flex: 1,
    position: 'absolute',
    width: '100%',
    height: '100%',
  },

  login: {
    height: 65,
    backgroundColor: COLORS.red,
  },
  signup: {
    height: 65,
    backgroundColor: COLORS.green,
  },
});

export default WelcomeScreen;
