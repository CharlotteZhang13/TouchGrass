import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

const Timer = ({formattedTime}) => {
  return (
    <View style={styles.timeContainer}>
      <Text style={styles.timeText}>
        {formattedTime}
      </Text>
    </View>
  );
};

const styles = StyleSheet.create({
  timeContainer: {
  },
  timeText: {
    color: '#F3705F', // Blue text color
    fontSize: 24, // Font size
    fontFamily: 'Inter_400Regular',
  },
});

export default Timer;
