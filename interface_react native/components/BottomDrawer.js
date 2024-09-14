// BottomDrawer.js
import React, { useRef } from 'react';
import {
  StyleSheet,
  View,
  Animated,
  PanResponder,
  Dimensions,
  FlatList,
  Image,
  Text,
} from 'react-native';

const { height: screenHeight, width: screenWidth } = Dimensions.get('window');
const imageGap = 10; // Gap between images (10px)
const numColumns = 3; // 3 columns
const imageSize = (screenWidth - (numColumns + 1) * imageGap) / numColumns; // Calculate image size

// Array of placeholder image URIs
const imagesArray = [
  { id: '1', uri: 'https://plus.unsplash.com/premium_photo-1667423049497-291580083466?w=800&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MXx8Z3Jhc3N8ZW58MHx8MHx8fDA%3D' },
  { id: '2', uri: 'https://images.unsplash.com/photo-1533460004989-cef01064af7e?w=800&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8Z3Jhc3N8ZW58MHx8MHx8fDA%3D' },
  { id: '3', uri: 'https://images.unsplash.com/photo-1560153203-67592026e61f?w=800&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8M3x8Z3Jhc3N8ZW58MHx8MHx8fDA%3D' },
  { id: '4', uri: 'https://images.unsplash.com/photo-1676070310591-db5d12c24680?q=80&w=2687&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D' },
  { id: '5', uri: 'https://plus.unsplash.com/premium_photo-1675802756329-145bac875bf7?q=80&w=2672&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D' },
  { id: '6', uri: 'https://images.unsplash.com/photo-1610571648632-7500e6e7c0e7?w=800&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Nnx8Z3Jhc3N8ZW58MHx8MHx8fDA%3D' },
  { id: '7', uri: 'https://images.unsplash.com/reserve/qstJZUtQ4uAjijbpLzbT_LO234824.JPG?q=80&w=2574&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D' },
  { id: '8', uri: 'https://images.unsplash.com/photo-1467740100611-36858db27485?q=80&w=2673&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D' },
  { id: '9', uri: 'https://images.unsplash.com/photo-1500285426772-410d8f2243d0?w=800&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTh8fGdyYXNzfGVufDB8fDB8fHww' },
  { id: '10', uri: 'https://images.unsplash.com/photo-1509219411165-3ec3195b4842?w=800&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTl8fGdyYXNzfGVufDB8fDB8fHww' },
  { id: '11', uri: 'https://plus.unsplash.com/premium_photo-1674061111250-7908c02b7878?w=800&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MjF8fGdyYXNzfGVufDB8fDB8fHww' },
  { id: '12', uri: 'https://images.unsplash.com/photo-1484538441406-ea31a113806f?w=800&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MjJ8fGdyYXNzfGVufDB8fDB8fHww' },
  { id: '13', uri: 'https://images.unsplash.com/photo-1492506014987-8dd98e591a7a?w=800&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MjR8fGdyYXNzfGVufDB8fDB8fHww' },
  { id: '14', uri: 'https://images.unsplash.com/photo-1413813447360-290ffe8d33d0?w=800&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mjd8fGdyYXNzfGVufDB8fDB8fHww' },
  { id: '15', uri: 'https://plus.unsplash.com/premium_photo-1664116928635-83695cb3da3c?w=800&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mjl8fGdyYXNzfGVufDB8fDB8fHww' },
  { id: '16', uri: 'https://images.unsplash.com/photo-1467073070072-6b91b7f4c848?w=800&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MzJ8fGdyYXNzfGVufDB8fDB8fHww' },
  { id: '17', uri: 'https://images.unsplash.com/photo-1528905857277-9c870593f2e9?w=800&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MzV8fGdyYXNzfGVufDB8fDB8fHww' },
  { id: '18', uri: 'https://images.unsplash.com/photo-1475140604894-b4aaba075542?w=800&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mzl8fGdyYXNzfGVufDB8fDB8fHww' }
];

const BottomDrawer = () => {
  const animatedValue = useRef(new Animated.Value(screenHeight - 100)).current; // Initially drawer is closed

  const panResponder = useRef(
    PanResponder.create({
      onMoveShouldSetPanResponder: (_, gestureState) => {
        return Math.abs(gestureState.dy) > 10;
      },
      onPanResponderMove: (_, gestureState) => {
        const newY = Math.max(
          0,
          Math.min(screenHeight - 100, animatedValue._value + gestureState.dy)
        );
        animatedValue.setValue(newY);
      },
      onPanResponderRelease: (_, gestureState) => {
        if (gestureState.dy > 0) {
          // Swiping down
          Animated.spring(animatedValue, {
            toValue: screenHeight - 100, // Slide it back down to the bottom
            useNativeDriver: false,
          }).start();
        } else {
          // Swiping up
          Animated.spring(animatedValue, {
            toValue: 100, // Slide it all the way to the top
            useNativeDriver: false,
          }).start();
        }
      },
    })
  ).current;

  const renderItem = ({ item }) => (
    <View style={styles.gridItem}>
      <Image source={{ uri: item.uri }} style={styles.image} />
    </View>
  );

  return (
    <Animated.View
      style={[
        styles.drawer,
        {
          transform: [{ translateY: animatedValue }],
        },
      ]}
      {...panResponder.panHandlers}>
      <View style={styles.drawerHandle} />
      <View style={styles.drawerContent}>
        {/* FlatList to render images in 3-column grid */}
          <Text style={styles.drawerText}>Gallery</Text>
        <FlatList
          data={imagesArray}
          renderItem={renderItem}
          keyExtractor={(item) => item.id}
          numColumns={numColumns} // 3 columns
          contentContainerStyle={styles.grid}
        />
      </View>
    </Animated.View>
  );
};

const styles = StyleSheet.create({
  drawer: {
    position: 'absolute',
    width: '100%',
    height: screenHeight, // Full screen height
    bottom: 0,
    backgroundColor: 'white',
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 5 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
    elevation: 5,
  },
  drawerHandle: {
    width: 40,
    height: 5,
    borderRadius: 2.5,
    backgroundColor: '#ccc',
    alignSelf: 'center',
    marginVertical: 10,
  },
  drawerContent: {
    display: 'flex',
    flexDirection: 'column',
    gap: 20,
    flex: 1,
    paddingHorizontal: 10,
    paddingTop: 10,
  },
  grid: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  gridItem: {
    width: imageSize,
    height: imageSize,
    margin: imageGap / 2, // Add half the gap to each side of the image
  },
  image: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
    borderRadius: 10,
  },
    drawerText: {
    fontSize: 32,
    fontFamily: 'Inter_700Bold',
    color: '#3564C8',
    textAlign: 'center'
  }
});

export default BottomDrawer;
