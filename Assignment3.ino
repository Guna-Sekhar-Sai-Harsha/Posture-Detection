
#include <Arduino_LSM9DS1.h>
  
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

#include "model.h"
#define DEBUG 1
const int numSamples = 238;

int samplesRead = numSamples;

// global variables used for TensorFlow Lite (Micro)
tflite::MicroErrorReporter tflErrorReporter;

// pull in all the TFLM ops, you can remove this line and
// only pull in the TFLM ops you need, if would like to reduce
// the compiled size of the sketch.
tflite::AllOpsResolver tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// Create a static memory buffer for TFLM, the size may need to
// be adjusted based on the model you are using
constexpr int tensorArenaSize = 48 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

// array to map position index to a name
const char* Position[] = {
  "Supine",
  "Prine",
  "Side",
  "Sitting",
  "Unknown"
};

#define NUM_Positions (sizeof(Position) / sizeof(Position[0]))

void setup() {
  Serial.begin(9600);
  while (!Serial);

  // initialize the IMU
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  // print out the samples rates of the IMUs
  Serial.print("Accelerometer sample rate = ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println(" Hz");
  Serial.print("Gyroscope sample rate = ");
  Serial.print(IMU.gyroscopeSampleRate());
  Serial.println(" Hz");

  Serial.println();

  // get the TFL representation of the model byte array

  tflModel = tflite::GetModel(g_model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  // Create an interpreter to run the model
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);

  // Allocate memory for the model's input and output tensors
  tflInterpreter->AllocateTensors();

  // Get pointers for the model's input and output tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
 #if DEBUG
  Serial.print("Number of dimensions: ");
  Serial.println(tflInputTensor->dims->size);
  Serial.print("Dim 1 size: ");
  Serial.println(tflInputTensor->dims->data[0]);
  Serial.print("Dim 2 size: ");
  Serial.println(tflInputTensor->dims->data[1]);
  Serial.print("Dim 3 size: ");
  Serial.println(tflInputTensor->dims->data[2]);
  Serial.print("Input type: ");
  Serial.println(tflInputTensor->type);
 #endif
}

void loop() {
  float x,y,z;
  int sensor;
  if (IMU.accelerationAvailable()) {
    if (IMU.gyroscopeAvailable()) {
      if (IMU.magneticFieldAvailable()) {
          
          if(Serial.available()>0){
            sensor= Serial.read();
          
          switch(sensor){
            case 0: IMU.readAcceleration(x, y, z);
                    break;
            case 1: IMU.readGyroscope(x, y, z);
                    break;
            case 2: IMU.readMagneticField(x, y, z);
                    break;
          }
      Serial.print("x:");
      Serial.print(x);
      Serial.print("y:");
      Serial.print(y);
      Serial.print("z:");
      Serial.print(z);
      Serial.print("Samples Read:");
      Serial.println(samplesRead);
  // check if the all the required samples have been read since
  // the last time the significant motion was detected
  while (samplesRead < numSamples) {
      // normalize the IMU data between 0 to 1 and store in the model's
      // input tensor
      tflInputTensor->data.f[samplesRead * 3 + 0] = x;
      tflInputTensor->data.f[samplesRead * 3 + 1] = y;
      tflInputTensor->data.f[samplesRead * 3 + 2] = z;
      samplesRead++;

      if (samplesRead == numSamples) {
        // Run inferencing
        TfLiteStatus invokeStatus = tflInterpreter->Invoke();
        if (invokeStatus != kTfLiteOk) {
          Serial.println("Invoke failed!");
          while (1);
          return;
        }

        // Loop through the output tensor values from the model
        for (int i = 0; i < NUM_Positions; i++) {
          Serial.print(Position[i]);
          Serial.print(": ");
          Serial.println(tflOutputTensor->data.f[i], 3);
        }
        Serial.println();
        delay(10);
      }
    }
  }
}
}
}
}
