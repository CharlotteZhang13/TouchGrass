import React from 'react';
import { View, Text, Dimensions, StyleSheet } from 'react-native';
import Svg, { Circle } from 'react-native-svg';

const screenWidth = Dimensions.get('window').width;

const CloudWithText = ({numberTouches}) => {
  return (
    <View style={styles.cloudWithTextContainer}>
      {/* Cloud SVG */}
      <Svg height="300" width={screenWidth} viewBox={`0 0 ${screenWidth} 300`}>
        {/* Left puff */}
        <Circle cx={screenWidth * 0.2} cy="100" r="100" fill="white" />

        {/* Center puff, positioned lower */}
        <Circle cx={screenWidth * 0.5} cy="150" r="130" fill="white" />

        {/* Right puff */}
        <Circle cx={screenWidth * 0.8} cy="100" r="100" fill="white" />
      </Svg>
      <Text style={styles.textInCloud}>
          <Text style={styles.blueText}>I touched grass</Text>{"\n"}
          <Text style={styles.greenText}>{numberTouches} times</Text>{"\n"}
          <Text style={styles.blueText}>today</Text>
        </Text>
    </View>
  );
};

const styles = StyleSheet.create({
  cloudWithTextContainer: {
    position: 'relative',
    top: -20,
    width: '100%',
    alignItems: 'center', // Centers the text horizontally
  },
  textInCloud: {
    position: 'absolute',
    top: 80, // Adjust this based on where you want the text vertically within the cloud
    textAlign: 'center', // Centers the text inside the View
    fontFamily: 'Inter_700Bold',  // Use Inter Regular
    fontSize: 32,
    lineHeight: 48,
  },
  blueText: {
    color: '#3564C8',
  },
  greenText: {
    color: '#30DD92',
  },
});

export default CloudWithText;
