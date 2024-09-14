import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  Button,
  StyleSheet,
  Image,
  Platform,
  TouchableOpacity,
  ActivityIndicator,
  Animated,
} from 'react-native';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';
import { decodeJpeg } from '@tensorflow/tfjs-react-native';
import * as ImagePicker from 'expo-image-picker';
import * as FileSystem from 'expo-file-system';
import { MaterialIcons } from '@expo/vector-icons';

export default function App() {
  const [model, setModel] = useState(null);
  const [loading, setLoading] = useState(true);
  const [prediction, setPrediction] = useState(null);
  const [imageUri, setImageUri] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [showCelebration, setShowCelebration] = useState(false);
  const [showWarning, setShowWarning] = useState(false);
  const fadeAnim = useState(new Animated.Value(0))[0];

  const imageUrls = [
    'https://th.bing.com/th/id/OIP.x--Qz4d68wGjUD411gAnEgAAAA?rs=1&pid=ImgDetMain',
    'https://th.bing.com/th/id/OIP.y1LAiETgF2mLIWSNxpmk5wHaFj?w=222&h=180&c=7&r=0&o=5&dpr=2.5&pid=1.7',
    'https://th.bing.com/th/id/OIP.8qo_pge6nNWj_imUcBpOLAHaEQ?w=310&h=180&c=7&r=0&o=5&dpr=2.5&pid=1.7',
    'https://th.bing.com/th/id/OIP.8DQ2oMcUvtI0yG914oA32AHaE7?w=281&h=187&c=7&r=0&o=5&dpr=2.5&pid=1.7',
    'https://th.bing.com/th/id/OIP.jsmNHbVwU4hcGGic_OlxxwHaFs?rs=1&pid=ImgDetMain',
    'https://media.istockphoto.com/photos/wide-angle-shot-of-students-and-youth-in-lecture-hall-in-east-asia-picture-id1150261801?k=20&m=1150261801&s=612x612&w=0&h=HETIv04umJUb6_d3tpNERzbVWiCZBgn7xPkK_dOdNzg=',
  ];

  useEffect(() => {
    async function loadModel() {
      try {
        await tf.ready();
        const modelUrl = 'https://CharlotteZhang13.github.io/tfjs/model.json';
        const loadedModel = await tf.loadLayersModel(modelUrl);
        setModel(loadedModel);
        setLoading(false);
      } catch (error) {
        console.error('Error loading TensorFlow.js model:', error);
      }
    }
    loadModel();
  }, []);

  const pickImage = async () => {
    // Ask for permission to access the camera
    const permissionResult = await ImagePicker.requestCameraPermissionsAsync();

    if (!permissionResult.granted) {
      alert('Permission to access camera is required!');
      return;
    }

    // Open camera to take a photo
    let result = await ImagePicker.launchCameraAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.canceled && result.assets && result.assets.length > 0) {
      const imageUri = result.assets[0].uri; // Correctly access the URI from assets
      setImageUri(imageUri);
      classifyLocalImage(imageUri);
    } else {
      console.error('No image selected or image picker was canceled');
    }
  };

  const classifyLocalImage = async (uri) => {
    try {
      setIsLoading(true);
      // Read the image file as a binary string
      const imgB64 = await FileSystem.readAsStringAsync(uri, {
        encoding: FileSystem.EncodingType.Base64,
      });

      // Convert Base64 image to Uint8Array for TensorFlow.js processing
      const imageBuffer = tf.util.encodeString(imgB64, 'base64').buffer;
      const uint8Array = new Uint8Array(imageBuffer);

      // Decode the JPEG image to a tensor
      const imageTensor = decodeJpeg(uint8Array);

      // Preprocess the image (resize and normalize as needed)
      const resizedImage = tf.image.resizeBilinear(imageTensor, [180, 180]); // Adjust size as per your model

      // Make prediction
      const predictions = await model.predict(resizedImage.expandDims(0));

      // Get top prediction (adjust based on your model's output format)
      const topPrediction = predictions.argMax(-1).dataSync()[0];
      setPrediction(
        topPrediction === 0 ? 'You touched grass!' : 'This is not grass :('
      );
      setIsLoading(false);
      if (topPrediction === 0) {
        setShowCelebration(true);
        Animated.timing(fadeAnim, {
          toValue: 1,
          duration: 2000,
          useNativeDriver: true,
        }).start(() => {
          // Hide the celebration effect after the animation completes
          setTimeout(() => {
            setShowCelebration(false);
            fadeAnim.setValue(0); // Reset fadeAnim value
          }, 2000); // Matches the duration of the fade-in animation
        });
      } else {
        setShowWarning(true);
        Animated.timing(fadeAnim, {
          toValue: 1,
          duration: 2000,
          useNativeDriver: true,
        }).start(() => {
          // Hide the celebration effect after the animation completes
          setTimeout(() => {
            setShowWarning(false);
            fadeAnim.setValue(0); // Reset fadeAnim value
          }, 2000); // Matches the duration of the fade-in animation
        });
      }
    } catch (error) {
      console.error('Error classifying image:', error);
      setIsLoading(false);
    }
  };

  const classifyRemoteImage = async (uri) => {
    if (model) {
      try {
        setIsLoading(true);
        // Fetch image as blob
        const response = await fetch(uri);

        const blob = await response.blob();
        const reader = new FileReader();
        reader.readAsArrayBuffer(blob);
        reader.onloadend = async () => {
          const arrayBuffer = reader.result;
          const uint8Array = new Uint8Array(arrayBuffer);

          const imageTensor = decodeJpeg(uint8Array);

          const resizedImage = tf.image.resizeBilinear(imageTensor, [180, 180]); // Adjust size as per your model
          const predictions = await model.predict(resizedImage.expandDims(0));

          const topPrediction = predictions.argMax(-1).dataSync()[0];
          setPrediction(
            topPrediction === 0 ? 'You touched grass!' : 'This is not grass :('
          );
          setIsLoading(false);
      if (topPrediction === 0) {
        setShowCelebration(true);
        Animated.timing(fadeAnim, {
          toValue: 1,
          duration: 2000,
          useNativeDriver: true,
        }).start(() => {
          // Hide the celebration effect after the animation completes
          setTimeout(() => {
            setShowCelebration(false);
            fadeAnim.setValue(0); // Reset fadeAnim value
          }, 2000); // Matches the duration of the fade-in animation
        });
      } else {
        setShowWarning(true);
        Animated.timing(fadeAnim, {
          toValue: 1,
          duration: 2000,
          useNativeDriver: true,
        }).start(() => {
          // Hide the celebration effect after the animation completes
          setTimeout(() => {
            setShowWarning(false);
            fadeAnim.setValue(0); // Reset fadeAnim value
          }, 2000); // Matches the duration of the fade-in animation
        });
      }
        };
      } catch (error) {
        setIsLoading(false);
        console.error('Error classifying image:', error);
      }
    }
  };

  const handleImagePress = async (url) => {
    setImageUri(url);
    classifyRemoteImage(url);
  };

  return (
    <View style={styles.container}>
      {loading ? (
        <Text>Loading model...</Text>
      ) : (
        <>
          {imageUri && (
            <Image source={{ uri: imageUri }} style={styles.image} />
          )}
          <Button title="Take a photo" onPress={pickImage} />
          {prediction && <Text>{prediction}</Text>}
        </>
      )}
      <View style={styles.gridContainer}>
        {imageUrls.map((url, index) => (
          <TouchableOpacity key={index} onPress={() => handleImagePress(url)}>
            <Image source={{ uri: url }} style={styles.gridImage} />
          </TouchableOpacity>
        ))}
      </View>
      {isLoading && (
        <View style={styles.overlay}>
          <ActivityIndicator size="large" color="#0000ff" />
        </View>
      )}

      {showCelebration && (
        <Animated.View style={[styles.celebrationOverlay, { opacity: fadeAnim }]}>
          <MaterialIcons name="celebration" size={100} color="green" />
          <Text style={styles.celebrationText}>YOU TOUCHED GRASS !!! </Text>
        </Animated.View>
      )}

      {showWarning && (
        <Animated.View style={[styles.warningOverlay, { opacity: fadeAnim }]}>
          <MaterialIcons name="warning" size={100} color="red" />
          <Text style={styles.warningText}>Retry, it is not grass </Text>
        </Animated.View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#ecf0f1',
  },
  image: {
    width: 200,
    height: 200,
    marginTop: 20,
  },
  gridContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'center',
    marginTop: 20,
    marginBottom: 20,
    width: '100%',
    maxWidth: 600,
  },
  gridImage: {
    width: 100,
    height: 100,
    margin: 5,
  },
  overlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0, 0, 0, 0.5)', // Gray out effect
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 1000,
  },
  celebrationOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(255, 255, 255, 0.5)', // Semi-transparent overlay for celebration effect
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 1000,
  },
  celebrationText: {
    fontSize: 24,
    color: 'green',
    marginTop: 10,
  },
  warningOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(255, 255, 255, 0.5)', // Semi-transparent overlay for celebration effect
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 1000,
  },
  warningText: {
    fontSize: 24,
    color: 'red',
    marginTop: 10,
  },
});
