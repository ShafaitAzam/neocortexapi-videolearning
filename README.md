# Test And Investigation Of Video Learning Project With NeoCortexApi:

## 1. Motivation:
The Project "Video Learning With HTM CLA" is built upon the concept of Brain Theory that how a specific portion of our brain which is neocortex can learn multiple sequences of videos and can successfully predict the next frame from a given input frame of a video. The whole project is built on the basis of [NeoCortex Api](https://github.com/ddobric/neocortexapi). It uses Hierarchical Themporal Memory with Cortical Learning Algorithm as a learning mechanism. As we know HTM systems work with the data which is nothing but streams of zeros and ones(Binary Data). So the video is converted into the sequences of bit arrary and then it has been pushed to the HTM model to learn significant patterns from the video. Afterwards, when the model is ready for test phase, an arbitrary image is passed through the network and the model then tries to recreate the video by predicting the proceeding frame from the input frame(arbitrary image).

## 2. Overview:
The overall project can be found in the [Project Folder](https://github.com/ddobric/neocortexapi/tree/SequenceLearning_ToanTruong/Project12_HTMCLAVideoLearning/HTMVideoLearning). This project contemplates the existing Sequence Learning procedure in [SequenceLearning.cs](https://github.com/ddobric/neocortexapi/tree/master/source/Samples/NeoCortexApiSample). WIth the help of python libraries the input training sets have been generated. [DataGeneration](https://github.com/ddobric/neocortexapi/tree/SequenceLearning_ToanTruong/DataGeneration). Three videos have been created with moving circle, triangle and rectangle. The program tries to read videos as input and for that a library has been created using OpenCV2. To run the project in the C# platform this [VideoLibrary](https://github.com/ddobric/neocortexapi/tree/SequenceLearning_ToanTruong/Project12_HTMCLAVideoLearning/HTMVideoLearning/VideoLibrary) needed some nuget packages named [Emgu.CV](https://www.nuget.org/packages/Emgu.CV/), [Emgu.CV.Bitmap](https://www.nuget.org/packages/Emgu.CV.Bitmap/) and [Emgu.CV.runtimes.windows](https://www.nuget.org/packages/Emgu.CV.runtime.windows/) with version higher than 4.5.3. The project reads input videos defined in the path and convert each videos into sequence of bit arrays. These bit arrays then pushed to the Spatial pooler where the learning stage begins. Homeostatic Plasticity Controller is connected with the SP so that it can reach to the stable state without forgetting. Afterwards the Temporal memory is introduced and the learning procedure is continued with SP and TM. The testing section has also been added where prediction is done from an input frame and the corresponding video has been generated depending on the correct prediction.

Current encoding mechanism of the frames allows the conversion of each pixels into a portion in input bit array which is then pushed to the HTM model for training purpose. There are three sets of video in the VideoLibrary/SmallTrainingSet which are all in black and white(Circle, Triangle and Rectangle). These are all small size videos containing one label for each video. For testing and experimental purpose we have created another three small videos which contains label named Pentagon, Star and Hexagon accordingly. These newly created data sets are all in black and white color mode. Currently, we are trying to analyse accuracy of the model when it is being tested with these newly created data sets after training. Now we are creating videos with PURE color mode. Afterwards we will start training the existing model with BINARIZEDRGB and PURE color mode videos.
  

This project references Sequence Learning sample, see [SequenceLearning.cs](https://github.com/ddobric/neocortexapi/tree/master/source/Samples/NeoCortexApiSample).  

Input Videos are currently generated from python scripts, using OpenCV2. See [DataGeneration](https://github.com/ddobric/neocortexapi/tree/SequenceLearning_ToanTruong/DataGeneration) for manual on usage and modification.  

The Reading of Videos are enabled by [VideoLibrary](https://github.com/ddobric/neocortexapi/tree/SequenceLearning_ToanTruong/Project12_HTMCLAVideoLearning/HTMVideoLearning/VideoLibrary), written for this project using OpenCV2. This library requires nuget package [Emgu.CV](https://www.nuget.org/packages/Emgu.CV/), [Emgu.CV.Bitmap](https://www.nuget.org/packages/Emgu.CV.Bitmap/), [Emgu.CV.runtimes.windows](https://www.nuget.org/packages/Emgu.CV.runtime.windows/) version > 4.5.3.  

Learning process include: 
1. reading videos.
2. convert videos to Lists of bitarrays.
3. Spatial Pooler Learning with Homeostatic Plasticity Controller until reaching stable state.
4. Learning with Spatial pooler and Temporal memory, conditional exit.
5. Interactive testing section, output video from frame input.
## 3. Data Generation:
The current encoding mechanism of the frame employs the convert of each pixels into an part in the input bit array. This input bit array is used by the model for training.  
There are currently 3 training set:
- SmallTrainingSet: has 3 video, 1 foreach label in {circle rectangle triangle}.    
- TrainingVideos: has more video, intended for training in `PURE` colorMode
- oneVideoTrainingSet
The current most used set for training and debugging is SmallTrainingSet. Also new videos have been generated for testing purpose(pentagon, hexagon and star). Different techniques have been used to generate these new video set which can be found here [NewDataGeneration](https://github.com/ShafaitAzam/neocortexapi-videolearning/blob/Shafait_Azam_CodeX/DataGeneration/DataGeneration.docx)  

## 4. Videos Reading:
For the purpose of encoding and reading video data from the SmallTrainingSet a library has been created. This library consists of three subclasses named **VideoSet**, **NVideo** and **NFrame**. In the following section the detail analysis of these classes has been added.

- [**1. VideoSet**](https://github.com/ddobric/neocortexapi/blob/SequenceLearning_ToanTruong/Project12_HTMCLAVideoLearning/HTMVideoLearning/VideoLibrary/VideoSet.cs):
The following code section in the VideoSet class reads video from the video folder path and the label has been added in accordance with the folder name.

```csharp
public VideoSet(string videoSetPath, ColorMode colorMode, int frameWidth, int frameHeight, double frameRate = 0)
        {
            nVideoList = new List<NVideo>();
            Name = new List<string>();
            // Set the label of the video collection as the name of the folder that contains it 
            this.VideoSetLabel = Path.GetFileNameWithoutExtension(videoSetPath);

            // Read videos from the video folder path 
            nVideoList = ReadVideos(videoSetPath, colorMode, frameWidth, frameHeight, frameRate);
        }
 ```
This class also defines a function named CreateConvertedVideos which is used to create a video with defined frameRate, frameWidth and frameHeight and put that video in a folder. Afterwards, a folder has been created which consists nothing but the converted frames in png format. These images (frames) can be used for the prediction purpose.

```csharp
public void CreateConvertedVideos(string videoOutputDirectory)
        {
            foreach (NVideo nv in nVideoList)
            {
                string folderName = $"{videoOutputDirectory}" + @"\" + $"{nv.label}";
                if (!Directory.Exists(folderName))
                {
                    Directory.CreateDirectory(folderName);
                }
                NVideo.NFrameListToVideo(nv.nFrames, $"{folderName}" + @"\" + $"{nv.name}", (int)nv.frameRate, new Size(nv.frameWidth, nv.frameHeight), true);
                if (!Directory.Exists($"{folderName}" + @"\" + $"{nv.name}"))
                {
                    Directory.CreateDirectory($"{folderName}" + @"\" + $"{nv.name}");
                }
                for (int i = 0; i < nv.nFrames.Count; i += 1)
                {
                    nv.nFrames[i].SaveFrame($"{folderName}" + @"\" + $"{nv.name}" + @"\" + $"{nv.nFrames[i].FrameKey}.png");
                }
            }
        }
```

- [**2. NVideo**](https://github.com/ddobric/neocortexapi/blob/SequenceLearning_ToanTruong/Project12_HTMCLAVideoLearning/HTMVideoLearning/VideoLibrary/NVideo.cs):
This class has been defined to read a video in different frame rates, frame height , frame width and color modes. The following code segment will show the basic settings of the parameters to manupulate a video.

```csharp
// Video Parameter 
            int frameWidth = 18;
            int frameHeight = 18;
            ColorMode colorMode = ColorMode.BLACKWHITE;
            // frame rate of 10 or smaller is possible
            double frameRate = 10;
 ```
For the experimental purpose the frameWidth and frameHeight parameters are chaged to 20, 22 and 24 to test the various segments of the existing project. The NFrameListToVideo function is used to compress and write a video after the prediction part had been done which returns the calculated predicted frame and all other frames that come after this frame. The VideoWriter function in the following code segments is used to compress the video in the mp4 format. Also in the parameter,-1 which automatically defines a codec for the purpose compression. Here, fourcc = -1. fourcc means 4-character code of codec to compress the frames. For example, VideoWriter::fourcc('P','I','M','1') is a MPEG-1 codec, VideoWriter::fourcc('M','J','P','G') is a motion-jpeg codec etc.

```csharp
public static void NFrameListToVideo(List<NFrame> bitmapList, string videoOutputPath, int frameRate, Size dimension, bool isColor)
        {
            using (VideoWriter videoWriter = new($"{videoOutputPath}.mp4", -1, (int)frameRate, dimension, isColor))
            {
                foreach (NFrame frame in bitmapList)
                {
                    Bitmap tempBitmap = frame.IntArrayToBitmap(frame.EncodedBitArray);
                    videoWriter.Write(tempBitmap.ToMat());
                }
            }
        }
```

- [**NFrame**](https://github.com/ddobric/neocortexapi/blob/SequenceLearning_ToanTruong/Project12_HTMCLAVideoLearning/HTMVideoLearning/VideoLibrary/NFrame.cs):
In this class different color modes have been defined such as BLACKWHITE, BINARIZEDRGB and PURE. The resolution of the actual video has been reduced for the purpose of testing. With the help of this class each bitmap has been encoded to an int array and binary array by iterating through every pixel of the frame. The function can also convert a bit array into bitmap.The Frame key parameter in this class is used to index the frames which is defined by **Framkey = (label)\_(VideoName)\_(index)**. This frame key is used for the classification purpose in the [HTMClassifier](https://github.com/ddobric/neocortexapi/blob/master/source/NeoCortexApi/Classifiers/HtmClassifier.cs).

**NOTE:**  
The current implementation of VideoLibrary saves all training data into a List of VideoSet, which contains all video information and their contents. For further scaling of the training set. It would be better to only store the index, where to access the video from the training data. This way the data would only be access when it is indexed and save memory for other processes.
## 5. Learning Process:
The following code segment depicts the configuration of the Hierarchical Themporal Memory Model that have been designed to learn video. Run1() and Run2() uses same configuration. I have tested some combination of these parameters especially the PermanenceDecrement and the PermanenceIncrement parameter. It takes significantly long time to train the model with the given input set(HTMVideoLearning\VideoLibrary\SmallTrainingSet). As these two parameters are responsible for manupulating the overlap connections in the SP column, these parameters changes the accuracy of the model. I have also analyzed the [SequenceLearningExperiment](https://github.com/ddobric/neocortexapi/blob/master/source/SequenceLearningExperiment/Program.cs) and compare it with our project. Though we are learning videos, I found that above two parameters are essential when we want to learn sequences of numbers. Punishing of segments has not been initialized here to drop the connection of SP given a particular input space.
The following code segment depicts the configuration of the Hierarchical Themporal Memory Model that have been designed to learn video. Run1() and Run2() uses same configuration. I have tested some combination of these parameters especially the PermanenceDecrement and the PermanenceIncrement parameter. It takes significantly long time to train the model with the given input set(HTMVideoLearning\VideoLibrary\SmallTrainingSet). As these two parameters are responsible for manupulating the overlap connections in the SP column, these parameters changes the accuracy of the model. I have also analyzed the SequenceLearningExperiment [Program.cs](https://github.com/ddobric/neocortexapi/blob/master/source/SequenceLearningExperiment/Program.cs) and compare it with our project. Though we are learning videos, I found that above two parameters are essential when we want to learn sequences of numbers. Punishing of segments has not been initialized here to drop the connection of SP given a particular input space.

```csharp
private static HtmConfig GetHTM(int[] inputBits, int[] numColumns)
{
    HtmConfig htm = new(inputBits, numColumns)
    {
        Random = new ThreadSafeRandom(42),
        CellsPerColumn = 30,
        GlobalInhibition = true,
        //LocalAreaDensity = -1,
        NumActiveColumnsPerInhArea = 0.02 * numColumns[0],
        PotentialRadius = (int)(0.15 * inputBits[0]),
        //InhibitionRadius = 15,
        MaxBoost = 10.0,
        //DutyCyclePeriod = 25,
        //MinPctOverlapDutyCycles = 0.75,
        MaxSynapsesPerSegment = (int)(0.02 * numColumns[0]),
        //ActivationThreshold = 15,
        //ConnectedPermanence = 0.5,
        // Learning is slower than forgetting in this case.
        //PermanenceDecrement = 0.15,
        //PermanenceIncrement = 0.15,
        // Used by punishing of segments.
    };
    return htm;
}
```
### 1. SP Learning with HomeoStatic Plasticity Controller (HPA):
This first section of learning use Homeostatic Plasticity Controller:
```csharp
HomeostaticPlasticityController hpa = new(mem, 30 * 150*3, (isStable, numPatterns, actColAvg, seenInputs) =>
{
    if (isStable)
        // Event should be fired when entering the stable state.
        Console.WriteLine($"STABLE: Patterns: {numPatterns}, Inputs: {seenInputs}, iteration: {seenInputs / numPatterns}");
    else
        // Ideal SP should never enter unstable state after stable state.
        Console.WriteLine($"INSTABLE: Patterns: {numPatterns}, Inputs: {seenInputs}, iteration: {seenInputs / numPatterns}");
        // We are not learning in instable state.
        learn = isInStableState = isStable;
        // Clear all learned patterns in the classifier.
        cls.ClearState();
}, numOfCyclesToWaitOnChange: 50);
```
The average number of cycles required for the "smallTrainingSet" is 270 cycles, "training with Spatial Pooler only" in this experiment used a while loop until the model reach true in parameter **IsInStableState**.
### 2. SP+TM Learning of frame sequences in the video set:
HPA will be triggered with the Compute method of the current layer. One problem during the learning is that even after successfull enter to stable state in Learning only with SP, the model can get unstable again after learning the first video or the second video in SP+TM stage. Thus:
```csharp
//Iteration:
foreach (VideoSet vd in videoData)
    {
    foreach (NVideo nv in vd.nVideoList)
        {
            // LOOP1
            // After finished learning in this cycle and move to the next video
            // The model somtimes becomes unstable and trigger cls.ClearState in HPA, making the HTMClassifier forget all what it has learned.  
            // Specificaly It clears the m_ActiveMap2 
            // To cope with this problem and faster debug the learning process, some time the experiment comment out the cls.ClearState() in HPA
        for (int i = 0; i < maxCycles; i++)
        }
    }
``` 
may be changed to:  
```csharp
//Iteration:
for (int i = 0; i < maxCycles; i++)
    {
        foreach (VideoSet vd in videoData)
        {
            foreach (NVideo nv in vd.nVideoList)
            {
                // LOOP2
                // This ensure the spreading learning of all the frames in different videos  
                // this keep cls.ClearState() in hpa and successully run to the end means that Learning process doesn't end in unstable state.
            }
        }
    }
``` 
For the current 2 tests:  
**_Run1: "SP only" runs LOOP2 || SP+TM runs LOOP1_**  
Key to learn with HTMClassifier: **FrameKey**, e.g.  rectangle_vd5_0, triangle_vd4_18, circle_vd1_9.  
Condition to get out from loop:
- Accuracy is calulated from prediction of all videos
- After run on each video, a loop is used to calculate the Predicted nextFrame of each frame in the video, the last frame doesn't have next predicted cells by usage of `tm.Reset(mem)` as indentification for end of video.  
```csharp
// correctlyPredictedFrame increase by 1 when the next FrameKey is in the Pool of n possibilities calculated from HTMClassifier cls.  
var lyrOut = layer1.Compute(currentFrame.EncodedBitArray, learn) as ComputeCycle;
var nextFramePossibilities = cls.GetPredictedInputValues(lyrOut.PredictiveCells.ToArray(), 1);
// e.g. if current frame is rectangle_vd5_0 and rectangle_vd5_1 is in nextFramePossibilities, then correctlyPredictedFrame for this Video increase by 1.  

double accuracy = correctlyPredictedFrame / ((double)trainingVideo.Count-1);
// The accuracy of each video add up to the final average accuracy of the VideoSet
videoAccuracy.Add(accuracy);
...
double currentSetAccuracy = videoAccuracy.Average();
// The accuracy of each VideoSet add up to the total cycleAccurary
setAccuracy.Add(currentSetAccuracy);
...
cycleAccuracy = setAccuracy.Average();
// The Learning is consider success when cycleAccuracy exceed 90% and stay the same more than 40 times
if(stableAccuracyCount >= 40 && cycleAccuracy> 0.9)
// The result is saved in Run1ExperimentOutput/TEST/saturatedAccuracyLog_Run1.txt.  
// In case the Experiment reach maxCycle instead of end condition, the log will be saved under Run1ExperimentOutput/TEST/MaxCycleReached.txt
``` 
After finishing the user will be prompted to input a picture path.  
The trained layer will use this image to try recreate the video it has learned from the training data.  
- The image can be drag into the command window and press enter to confirm input. The model use the input frame to predict the next frame, then continue the process with the output frame if there are still predicted cells from calculation. For the first prediction HTMClassifier takes at most 5 possibilities of the next frame from the input.  
- In case there are at least 1 frame, the codecs will appears and the green lines indicate the next predicted frame from the memory by HTMClassifier. 
- The output video can be found under Run1ExperimentOutput/TEST/ with the folder name (Predicted From "Image name").  
- Usually in this Test, The input image are chosen from the Directory Run1Experiment/converted/(label)/(videoName)/(FrameKey) for easier check if the trained model predict the correct next frame.  


**RESULT**  
- Due to the conversion of the input picture to fit the current model the input is also processed by VideoLibrary to the dimension of the training model. The scaled input image can also be found in the Run1ExperimentOutput/TEST/Predicted from (image name)/.  
- Ideally, a sequence of half the length of the video would regards this experiment as a success. Unfortunately, runs result in sequence of 1-5 frames after the input frame. 
- It is observed that the triangle set - the last training set in small training set has the best sequence generation with sometime up to 15 frames. 
- In some case frame that overlap each other e.g. the triangle at the same place of the circle may result in shape change but correct translation.  
- There are also cases where next frame calculated from input frame resulted in a loop sequence with 1,2,3 frame, these are bad connection and can continue to infinity.  
A max number of predicted frames was set to 42 to avoid running to infinity. The max number of frame in the current training set is 3*12 = 36 frames.
- This experiment has a very long run time because it covers all the frame in each cycle due to usage of LOOP1.

**_Run2: "SP only" runs LOOP2 || SP+TM runs LOOP2_**  
_This Run is inspired by the experiment [SequenceLearning_Duy](https://github.com/perfectaccountname/neocortexapi/blob/master/source/Samples/NeoCortexApiSample/SequenceLearning_DUY.cs), here the video would be the double sequence._  
Key to learn with HTMClassifier: **FrameKey**-**FrameKey**-...-**FrameKey**, e.g.  rectangle_vd5_0-rectangle_vd5_1-...rectangle_vd5_29.  

Run2 running used the following parameters:
1. previousInputs: the key used for learning with HTMClassifier(only at full length)
2. lastPredictedValue: the predicted value from the last cycle
3. maxPrevInputs = video length (in frame) -1

Input for the model is bit array created from encoding the current active frame.  
When starting the training for one video, the early cycle will begin to build up a list of all Framekey in time order.
```csharp
previousInputs.Add(currentFrame.FrameKey);
if (previousInputs.Count > (maxPrevInputs + 1))
    previousInputs.RemoveAt(0);

// In the pretrained SP with HPC, the TM will quickly learn cells for patterns
// In that case the starting sequence 4-5-6 might have the sam SDR as 1-2-3-4-5-6,
// Which will result in returning of 4-5-6 instead of 1-2-3-4-5-6.
// HtmClassifier allways return the first matching sequence. Because 4-5-6 will be as first
// memorized, it will match as the first one.
if (previousInputs.Count < maxPrevInputs)
    continue;
```
when previousInputs reach the same length of the video, the learning with HTMClassifier begins:  
```csharp
string key = GetKey(previousInputs);
List<Cell> actCells;
if (lyrOut.ActiveCells.Count == lyrOut.WinnerCells.Count)
{
    actCells = lyrOut.ActiveCells;
}
else
{
    actCells = lyrOut.WinnerCells;
}
cls.Learn(key, actCells.ToArray());
```
the key used for learning is generated from the FrameKey List previousInputs of the current Video.  
From this way of generating the key, the current frame bitArray Collums output will be learn with a key in which FrameKey sequence run from the next FrameKey from the current input to the end of the video and back to the current FrameKey.  
e.g. current frame: rectangle_vd5_4  
Video has 8 frames, then the generated key to be learned along with the active columns of current frame would be:  
_rectangle_vd5_5-rectangle_vd5_6-rectangle_vd5_7-rectangle_vd5_0-rectangle_vd5_1-rectangle_vd5_2-rectangle_vd5_3-rectangle_vd5_4_  

By Learning this way one frame info is associated with the information of the whole frame sequence (Video). This compared to Run1 reduce the error compared with predicting the frame one by one, the output video can also be recalled in full length. 

Condition to get out of the loop:  
- By using LOOP2, Run2 learn each video respectively. The training of one video ends after the model reach acuracy > 80% and stay the same for 50 cycles or reaching maxCycles. The models then trains with the next video until there is no video left.  
```csharp
if(accuracy == lastCycleAccuracy)
{
    // The learning may result in saturated accuracy
    // Unable to learn to higher accuracy, Exit
    saturatedAccuracyCount += 1;
    if (saturatedAccuracyCount >= 50 && lastCycleAccuracy>80)
        {
            ...
        }
}
```
In case the end condition is reached, a saturatedAccuracyLog will appear in Run2ExperimentOutput/TEST/.
After finishing the user will be prompted to input a picture path.  
The trained layer will use this image to try recreate the video it has learned from the training data.  
- The image can be drag into the command window and press enter to confirm input. The model use the input frame to predict the key.  
- if there are predicted cells from compute, HTMClassifier takes at most 5 possibilities (can be changed in code) of the predicted key.  
- In case there are at least 1 key, the codecs will appears and the green lines indicate the next predicted key from the memory by HTMClassifier. 
- The output video can be found under Run2ExperimentOutput/TEST/ with the folder name (Predicted From "Image name").  
- Usually in this Test, The input image are chosen from the Directory Run1Experiment/converted/(label)/(videoName)/(FrameKey) for easier check if the trained model predict the correct next frame.  

**RESULT**
- Due to the conversion of the input picture to fit the current model the input is also processed by VideoLibrary to the dimension of the training model. The scaled input image can also be found in the Run1ExperimentOutput/TEST/Predicted from (image name)/. 
- The log files after learning each video are also recorded as saturatedAccuracyLog_(label)_(video name) in the TEST/ directory.
- The output Video has full length of the video.
- The prediction sometimes forget the first video and enter unstable state again. This was mentioned in HPA above. The current way to cope with the phenomenom is comment out `cls.ClearState()` in declaration of HPA.
- Prediction ends with rather high accuracy 89-93% recorded after learning of a video.  
- the output video will run from the predicted next frame from input frame to the end of the video then back to the input frame.  

For an review on output folder TEST of both the Run after the learning one can refer to [SampleExperimentOutputTEST](https://github.com/ddobric/neocortexapi/tree/SequenceLearning_ToanTruong/Project12_HTMCLAVideoLearning/HTMVideoLearning/SampleExperimentOutputTEST).

Different videos have been generated in order to test the project in various conditions. Parameters of the code have been manipulated in order to figure out the impact of those parameters in the project. Detailed test reports have been generated which can be found here [TestReports](https://github.com/ShafaitAzam/neocortexapi-videolearning/tree/MySEProject/Documentation/TestReports). These includes test reports both for existing version of the project and the migrated version of the project.

