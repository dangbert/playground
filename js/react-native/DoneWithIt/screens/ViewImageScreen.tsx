import React from 'react';
import SafestAreaView from '../lib/SafestAreaView';
import { StyleSheet, Image, TouchableOpacity, View } from 'react-native';
import { COLORS } from '../constants';

interface ViewImageScreenProps {
  changeScreen: (next: string) => void;
}

const ViewImageScreen: React.FC<ViewImageScreenProps> = ({ changeScreen }) => {
  return (
    <SafestAreaView style={styles.container}>
      <View style={styles.controls}>
        <TouchableOpacity
          style={styles.close}
          onPress={() => changeScreen('welcome')}
        />
        <TouchableOpacity style={styles.delete} />
      </View>

      <View style={styles.imageContainer}>
        <Image style={styles.image} source={require('../assets/chair.jpg')} />
      </View>
    </SafestAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'black',
  },

  controls: {
    flexSize: 55,
    marginBottom: 12,
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginLeft: 30,
    marginRight: 30,
  },
  close: {
    height: 55,
    width: 55,
    backgroundColor: COLORS.red,
  },
  delete: {
    height: 55,
    width: 55,
    backgroundColor: COLORS.green,
  },

  imageContainer: {
    backgroundColor: 'inherit',
    flex: 1,
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    marginLeft: 12,
    marginRight: 12,
  },
  image: {
    width: '100%',
    height: '80%',
  },
});

export default ViewImageScreen;
