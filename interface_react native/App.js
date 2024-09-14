import React, {useState, useEffect} from 'react';
import { StyleSheet, Text, View, Image} from 'react-native';
import { useFonts, Inter_400Regular, Inter_700Bold, Inter_600SemiBold, Inter_500Medium } from '@expo-google-fonts/inter';
import AppLoading from 'expo-app-loading';
import { LinearGradient } from 'expo-linear-gradient';
import CloudWithText from './components/CloudWithText';
import Timer from './components/Timer';
import TouchGrassButton from './components/TouchGrassButton';
import BottomDrawer from './components/BottomDrawer';
import * as ImagePicker from 'expo-image-picker';  // Import the ImagePicker
const ONE_DAY_IN_MS = 5*24 * 60 * 60 * 1000; // 24 hours in milliseconds


const App = () => {
const [remainingTime, setRemainingTime] = useState(ONE_DAY_IN_MS); // Initialize timer with 24 hours
 const [selectedImage, setSelectedImage] = useState(null); // Store the captured image

  // Function to open the camera
  const openCamera = async () => {
    // Ask for permission to access the camera
    const { status } = await ImagePicker.requestCameraPermissionsAsync();

    if (status !== 'granted') {
      Alert.alert('Permission Denied', 'Sorry, we need camera permissions to make this work!');
      return;
    }

    // Launch the camera
    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    // Handle the captured image
    if (!result.cancelled) {
      setSelectedImage(result.uri); // Set the captured image URI
    }
  };

  const formatTime = (timeInMs) => {
    const totalSeconds = Math.floor(timeInMs / 1000);
    const days = Math.floor(totalSeconds/3600/24);
    const hours = Math.floor(totalSeconds/3600%24);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    return `${days}d ${hours}h ${minutes}m ${seconds}s`;
  };
    useEffect(() => {
    // Countdown function to update the timer every second
    const timerInterval = setInterval(() => {
      setRemainingTime((prevTime) => {
        if (prevTime <= 1000) {
          clearInterval(timerInterval); // Stop the timer when it reaches 0
          return 0;
        }
        return prevTime - 1000; // Decrease time by 1 second (1000 ms)
      });
    }, 1000);

    return () => clearInterval(timerInterval); // Clean up the interval on unmount
  }, []);
    const [touchCount, setTouchCount] = useState(0); // State to track the number of times button is pressed

  const handlePress = () => {
    setTouchCount(touchCount + 1); // Increment the count by 1 on press
    setRemainingTime(ONE_DAY_IN_MS); // Reset the timer to 24 hours
  };
  // Load the fonts
  const [fontsLoaded] = useFonts({
    Inter_400Regular,
    Inter_700Bold,
    Inter_600SemiBold,
    Inter_500Medium
  });

  if (!fontsLoaded) {
    return <AppLoading />;
  }

  return (
    <LinearGradient
      colors={['#87CEEB', 'white']}  // Gradient from blue to white
      style={styles.background}
    >
<Image source={require('./assets/images/avatar.png')} style = {styles.avatar} />
<Image style = {styles.cliff} source={require('./assets/images/cliff.png')}/>


      <CloudWithText numberTouches = {touchCount}/>
      
      {/* Correct use of style prop */}
      <View style = {styles.textWithTimer}>
      <Text style={styles.rockBottom}>
        Hit rock bottom in
      </Text>
      <Timer formattedTime = {formatTime(remainingTime)}/>
      </View>
            {/* Display the captured image */}
      {selectedImage && (
        <View style={styles.imageContainer}>
          <Image source={{ uri: selectedImage }} style={styles.capturedImage} />
        </View>
      )}
      <TouchGrassButton style = {styles.touchGrassButton} onPress = {handlePress, openCamera}/>
      <BottomDrawer/>
    </LinearGradient>
  );
};

const styles = StyleSheet.create({
  textWithTimer: {
    flexDirection: 'column',
    gap: 8,
    position: 'absolute',
    left: 24,
    top: '36%',
    alignItems: 'flex-start'
  },
  background: {
    flex: 1,
    justifyContent: 'flex-start',
    alignItems: 'center',
  },
  rockBottom: {
    position: 'relative',
    fontSize: 16,                 // Font size for the "Hit rock bottom in" text
    color: '#3564C8',                // Black color for the text
    fontFamily: 'Inter_600SemiBold',  // Use Inter Bold font for consistency
    textAlign: 'center',          // Center the text
  },
  avatar: {
    position: 'absolute',
    zIndex: 0,
    right: '10%',
    top: '30%',
  },
  cliff: {
    position: 'absolute',
    zIndex: -1,
    bottom: 0,
  }
});

export default App;
