import React from 'react';
import SafestAreaView from '../lib/SafestAreaView';
import { StyleSheet, Image, Text, TouchableOpacity, View } from 'react-native';

export default function WelcomeScreen() {
  return (
    <SafestAreaView style={styles.container}>
      <View style={styles.main}>
        <Image
          style={{ width: '100%', height: '100%' }}
          source={require('../assets/background.jpg')}
        />
      </View>

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
            style={{ width: 120, height: 120 }}
            source={require('../assets/logo-red.png')}
          />
          <Text style={{ fontSize: 20 }}>Sell What You Don't Need</Text>
        </View>

        <View>
          <TouchableOpacity style={styles.login}></TouchableOpacity>
          <TouchableOpacity style={styles.signup}></TouchableOpacity>
        </View>
      </View>
    </SafestAreaView>
  );
}

const RED = '#fc5c65';
const GREEN = '#4ECDC4';

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
    backgroundColor: RED,
  },
  signup: {
    height: 65,
    backgroundColor: GREEN,
  },
});
