// TouchGrassButton.js
import React from 'react';
import { StyleSheet, View, TouchableOpacity, Text } from 'react-native';
import MaterialIcons from 'react-native-vector-icons/MaterialIcons';

const TouchGrassButton = ({ onPress }) => {
  return (
    <View style={styles.container}>
    <View style={styles.tooltip}>
        <Text style={styles.tooltipText}>Touch grass</Text>
    </View>
    <TouchableOpacity onPress={onPress}>
      <View style={styles.circle}>
        <MaterialIcons name="touch-app" size={40} color="white" style = {styles.icon} />
      </View>
    </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    justifyContent: 'center',
    alignItems: 'center',
    position: 'absolute',
    bottom: '15%',
    left: 0,
  },
  circle: {
    backgroundColor: '#30DD92',
    width: 64,
    height: 64,
    borderRadius: 50,
    justifyContent: 'center',
    alignItems: 'center',
  },
  tooltip: {
    left: '80%',
    backgroundColor: 'white',
    borderRadius: 1000,
    padding: 8,
  },
  tooltipText: {
    color: '#3564C8',
    fontSize: 16,
    fontFamily: 'Inter_600SemiBold',
  },
  icon: {
    transform: [{ rotate: '180deg' }], // Rotating the text 45 degrees

  }
});

export default TouchGrassButton;
