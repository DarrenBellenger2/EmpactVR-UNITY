using UMA;
using UMA.PoseTools;

using System;
using System.Threading.Tasks;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEngine.UI;

using OpenCVForUnity.UnityUtils.Helper;
using OpenCVForUnity.UnityUtils;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.FaceModule;

using OpenCVForUnity.UtilsModule;
using OpenCVForUnityExample;
using OpenCVForUnity.VideoioModule;

using Stopwatch = System.Diagnostics.Stopwatch;

using UnityEngine.SceneManagement;

using Mediapipe;
using Mediapipe.Unity.CoordinateSystem;
using Mediapipe.Tasks.Vision.FaceLandmarker;
using UnityEngine.Rendering;
using Unity.Barracuda;

using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

using NWaves.Audio;
using NWaves.FeatureExtractors;
using NWaves.FeatureExtractors.Base;
using NWaves.FeatureExtractors.Multi;
using NWaves.FeatureExtractors.Options;
using NWaves.FeatureExtractors.Serializers;
using NWaves.Filters;
using NWaves.Filters.Base;
using NWaves.Filters.Fda;
using NWaves.Signals;
using NWaves.Transforms;
using NWaves.Windows;
	
namespace Mediapipe.Unity.Tutorial
{

	public class EmpactVideoUtilities : MonoBehaviour
	{
		
		public struct structAU {
			public float AU1, AU2, AU4, AU6, AU7, AU9, AU15, AU20, AU23, AU25, AU26;
		}
		structAU faceAU;
		public static List<structAU> overallFacialInput;
		public static List<structAU> facialInput;
		
		public Camera MainCam = null;
		[SerializeField] private TextAsset _configAsset;  
		[SerializeField] private RawImage _screen;		
		public UMAAvatarBase targetUMA = null;
		
		[Tooltip("Image ONNX ML model file")]
		public NNModel image_model;
		private Model m_ImageRuntimeModel;
		private IWorker image_worker;
		
		[Tooltip("Audio ONNX ML model file")]
		public NNModel audio_model;
		private Model m_AudioRuntimeModel;
		private IWorker audio_worker;		
		
		[Tooltip("Recognition intervals in seconds")]
		public float recognitionIntervalTime = 1;	
		public float recognitionIntervalFrames = 30;	
		//float recognitionTime = 0;
		float recognitionFrames = 0;			
		int Frequency = 44100; 
		int Cepstrum_length = 34;
		float audio_length; 	
		int audioBytesRead = 0;	

		int fftSize = 1024;
		double lowFreq = 100;     
		double highFreq = 8000; 
		int filterbankSize = 40;     
		
        ExpressionPlayer expressionPlayer;
        int counter = 0;
		
		//const int numFinalLandmarkPoints = 68;
		const int numFaceMeshLandmarkPoints = 468;
		const int numEmotions = 7;
		string[] observed_emotions = {"neutral", "happy", "sad", "anger", "fear", "disgust", "surprise"};
		
		// ==========================================================================
		bool refine_AU_extents = false;
		// ==========================================================================

        float AU1_low, AU1_high;
		float AU2_l_low, AU2_l_high; 
		float AU2_r_low, AU2_r_high;		
        float AU4_low, AU4_high;
		float AU6_low, AU6_high; 
					
		float AU7_l_low, AU7_l_high; 
		float AU7_r_low, AU7_r_high;
        float AU9_low, AU9_high;
		float AU15_l_low, AU15_l_high;
		float AU15_r_low, AU15_r_high;
		
		float AU20_low, AU20_high;		
		float AU23_low, AU23_high;
		float AU26_low, AU26_high; 			

		// ==========================================================================
		
		public Text Display_AU1;	
		public Text Display_AU2;	
		public Text Display_AU4;	
		public Text Display_AU6;	
		public Text Display_AU7;	
		public Text Display_AU9;	
		public Text Display_AU15;	
		public Text Display_AU20;	
		public Text Display_AU23;	
		public Text Display_AU25;	
		public Text Display_AU26;	
		
		public ProgressBar Progress_Happy;	
		public ProgressBar Progress_Surprise;
		public ProgressBar Progress_Neutral;
		public ProgressBar Progress_Sad;
		public ProgressBar Progress_Fear;
		public ProgressBar Progress_Anger;
		public ProgressBar Progress_Disgust;	

		public Text RecordButton_Text;	
		Boolean Start_Recording;
		Boolean Recording_FirstTime;
		
		private AudioSource audioSource;
		//temporary audio vector we write to every second while recording is enabled..
		List<float> tempRecording = new List<float>();
		//list of recorded clips...
		List<float[]> recordedClips = new List<float[]>();
		
		private UnityEngine.Rect screenRect;
		
		float row_average = 0f;
		List<float[]> imageBuffer = new List<float[]>();
		List<float[]> audioBuffer = new List<float[]>();
		float[] fusedBuffer = new float[numEmotions];
		float[] bufferTotals = new float[numEmotions];
		int frames_to_cut = 30;    // ADFES = 30    // RAVDESS = 15
		int num_images = 0;
		int buffer_index = 0;
		float ADSR_release = 1.5f;
		
		private CalculatorGraph _graph;
		private OutputStream<ImageFrame> _outputVideoStream;
		private OutputStream<List<NormalizedLandmarkList>> _multiFaceLandmarksStream;
		private ResourceManager _resourceManager;	
		
		private WebCamTexture _webCamTexture;
		private Texture2D _inputTexture;
		private Color32[] _inputPixelData;
		private Texture2D _outputTexture;
		private Color32[] _outputPixelData;
		private NormalizedLandmark test;	
		
		int currentFrame;
		int previousFrame;
		Stopwatch stopwatch;
		//private UnityEngine.Rect screenRect;
		
		int requestedMode = 0;
		bool video_capture_initialised = false;
		const int K_Webcam = 0;
		const int K_Video = 1;

		VideoCapture capture;
		bool isPlaying = false;
		bool shouldUpdateVideoFrame = false;
		long prevFrameTickCount;
		long currentFrameTickCount;	
		
		// ==============================================================================================================================
		// ==============================================================================================================================
		// RAVDESS ENLARGED
		// string[] observed_emotions = {"neutral", "happy", "sad", "anger", "fear", "disgust", "surprise"};

	//                             																							use_prev_landmarks     use_prev_landmarks    AUDIO			 	  		Overall
		//																														      False				    True								1s Intervals	Complete
		
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/RAVDESS_640_Labelled/neutral/Video_Man01.mp4"; //    Disgust				Disgust			Neutral		|		Disgust		Neutral
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/RAVDESS_640_Labelled/neutral/Video_Man02.mp4"; //    Neutral				Neutral			Neutral		|		Neutral		Neutral
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/RAVDESS_640_Labelled/neutral/Video_Man03.mp4"; //    Neutral				Neutral			Disgust		|		Neutral		Disgust
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/RAVDESS_640_Labelled/neutral/Video_Woman01.mp4"; //  Neutral				Neutral			Neutral		|		Neutral		Neutral		***
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/RAVDESS_640_Labelled/neutral/Video_Woman02.mp4"; //  Neutral				Neutral			Neutral		|		Neutral		Neutral		***	
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/RAVDESS_640_Labelled/neutral/Video_Woman03.mp4"; //  Fear					Neutral			Neutral		|		Fear		Neutral
		
		
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/RAVDESS_640_Labelled/happy/Video_Man01.mp4"; //      Happy					Happy			Happy		|		Happy		Happy		***
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/RAVDESS_640_Labelled/happy/Video_Man02.mp4"; // 	  Happy					Happy			Fear		|		Happy		Happy
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/RAVDESS_640_Labelled/happy/Video_Man03.mp4"; //      Happy					Happy			Happy		|		Happy		Happy		***
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/RAVDESS_640_Labelled/happy/Video_Woman01.mp4"; //    Happy					Happy			Happy		|		Happy		Happy
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/RAVDESS_640_Labelled/happy/Video_Woman02.mp4"; //    Happy					Happy			Surprise	|		Happy		Surprise
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/RAVDESS_640_Labelled/happy/Video_Woman03.mp4"; //    Happy					Happy			Happy		|		Happy		Happy		

		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/RAVDESS_640_Labelled/sad/Video_Man01.mp4"; // 		  Neutral				Sad        		Sad       	|		Neutral		Sad			***
		static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/RAVDESS_640_Labelled/sad/Video_Man02.mp4"; // 		  Disgust				Disgust			Sad			|		Sad!!!		Sad			***
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/RAVDESS_640_Labelled/sad/Video_Man03.mp4"; // 		  Neutral				Neutral			Surprise	|		Neutral		Surprise
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/RAVDESS_640_Labelled/sad/Video_Woman01.mp4"; // 	  Neutral				Neutral			Sad			|		Neutral		Neutral
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/RAVDESS_640_Labelled/sad/Video_Woman02.mp4"; //	  Neutral				Neutral			Disgust		|		Neutral		Disgust
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/RAVDESS_640_Labelled/sad/Video_Woman03.mp4"; // 	  Fear					Fear			Happy		|		Fear		Happy
		
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/RAVDESS_640_Labelled/anger/Video_Man01.mp4"; // 	  Anger					Anger			Anger		|		Anger		Anger		***
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/RAVDESS_640_Labelled/anger/Video_Man02.mp4"; // 	  Anger					Neutral			Anger		|		Anger		Anger		***
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/RAVDESS_640_Labelled/anger/Video_Man03.mp4"; // 	  Neutral				Neutral			Anger		|		Neutral		Anger		
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/RAVDESS_640_Labelled/anger/Video_Woman01.mp4"; // 	  Neutral				Neutral			Anger		|		Neutral		Anger
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/RAVDESS_640_Labelled/anger/Video_Woman02.mp4"; // 	  Neutral				Fear			Anger		|		Neutral		Anger
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/RAVDESS_640_Labelled/anger/Video_Woman03.mp4"; // 	  Surprise				Surprise		Happy		|		Surprise	Happy

		
		
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/RAVDESS_640_Labelled/disgust/Video_Man01.mp4"; // 	  Disgust				Disgust			Disgust		|		Disgust		Disgust		***
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/RAVDESS_640_Labelled/disgust/Video_Man02.mp4"; // 	  Disgust				Disgust			Disgust		|		Disgust		Disgust
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/RAVDESS_640_Labelled/disgust/Video_Man03.mp4"; // 	  Disgust				Disgust			Disgust		|		Disgust		Disgust
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/RAVDESS_640_Labelled/disgust/Video_Woman01.mp4"; //  Disgust				Disgust			Disgust		|		Disgust		Disgust
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/RAVDESS_640_Labelled/disgust/Video_Woman02.mp4"; //  Disgust				Disgust			Disgust		|		Disgust		Disgust		***
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/RAVDESS_640_Labelled/disgust/Video_Woman03.mp4"; // Fear					Disgust			Surprise	|		Fear		Surprise
		

		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/RAVDESS_640_Labelled/fear/Video_Man01.mp4"; // 	   Neutral				Neutral			Fear		|		Neutral		Fear
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/RAVDESS_640_Labelled/fear/Video_Man02.mp4"; // 	   Fear					Fear			Fear		|		Fear		Fear		***
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/RAVDESS_640_Labelled/fear/Video_Man03.mp4"; // 	   Fear					Fear			Surprise	|		Fear		Surprise
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/RAVDESS_640_Labelled/fear/Video_Woman01.mp4"; // 	   Fear					Fear			Surprise	|		Fear		Surprise
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/RAVDESS_640_Labelled/fear/Video_Woman02.mp4"; // 	   Fear					Fear			Fear		|		Fear		Fear
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/RAVDESS_640_Labelled/fear/Video_Woman03.mp4"; // 	   Fear					Fear			Fear		|		Fear		Fear		***

		
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/RAVDESS_640_Labelled/surprise/Video_Man01.mp4"; // 	Neutral				Neutral			Surprise	|		Neutral		Surprise	
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/RAVDESS_640_Labelled/surprise/Video_Man02.mp4"; // 	Neutral				Neutral			Surprise	|		Neutral		Surprise
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/RAVDESS_640_Labelled/surprise/Video_Man03.mp4"; // 	Fear				Fear			Surprise	|		Fear		Surprise	
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/RAVDESS_640_Labelled/surprise/Video_Woman01.mp4"; // 	Surprise			Surprise		Surprise	|		Surprise	Surprise	***
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/RAVDESS_640_Labelled/surprise/Video_Woman02.mp4"; // 	Fear				Fear			Surprise	|		Fear		Surprise
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/RAVDESS_640_Labelled/surprise/Video_Woman03.mp4"; // 	Surprise			Surprise		Surprise	|		Surprise	Surprise	***
		
		
		
		
		// ==============================================================================================================================
		// ==============================================================================================================================
		// ADFES
		
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/ADFES_Labelled_MP4/neutral/M01-Neutral-Face Forward-54.mp4"; // Surprise
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/ADFES_Labelled_MP4/neutral/M02-Neutral-Face Forward-57.mp4"; // Neutral
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/ADFES_Labelled_MP4/neutral/M03-Neutral-Face Forward-67.mp4"; // Neutral
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/ADFES_Labelled_MP4/neutral/F01-Neutral-Face Forward-7.mp4"; // Neutral
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/ADFES_Labelled_MP4/neutral/F02-Neutral-Face Forward-17.mp4"; // Sad
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/ADFES_Labelled_MP4/neutral/F03-Neutral-Face Forward-27.mp4"; // Neutral
		
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/ADFES_Labelled_MP4/happy/M01-Joy-Face Forward-53.mp4"; // Happy
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/ADFES_Labelled_MP4/happy/M02-Joy-Face Forward-56.mp4"; // Happy
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/ADFES_Labelled_MP4/happy/M03-Joy-Face Forward-66.mp4"; // Happy
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/ADFES_Labelled_MP4/happy/F01-Joy-Face Forward-6.mp4"; // Happy
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/ADFES_Labelled_MP4/happy/F02-Joy-Face Forward-16.mp4"; // Happy
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/ADFES_Labelled_MP4/happy/F03-Joy-Face Forward-26.mp4"; // Happy

		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/ADFES_Labelled_MP4/sad/M01-Sadness-Face Forward-56.mp4"; // Neutral
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/ADFES_Labelled_MP4/sad/M02-Sadness-Face Forward-59.mp4"; // Neutral
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/ADFES_Labelled_MP4/sad/M03-Sadness-Face Forward-69.mp4"; // Neutral
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/ADFES_Labelled_MP4/sad/F01-Sadness-Face Forward-9.mp4"; // Sad
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/ADFES_Labelled_MP4/sad/F02-Sadness-Face Forward-19.mp4"; // Neutral
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/ADFES_Labelled_MP4/sad/F03-Sadness-Face Forward-29.mp4"; // Neutral
		
		
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/ADFES_Labelled_MP4/anger/M01-Anger-Face Forward-48.mp4"; // Neutral
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/ADFES_Labelled_MP4/anger/M02-Anger-Face Forward-51.mp4"; // Neutral
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/ADFES_Labelled_MP4/anger/M03-Anger-Face Forward-61.mp4"; // Neutral
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/ADFES_Labelled_MP4/anger/F01-Anger-Face Forward-1.mp4"; // Anger
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/ADFES_Labelled_MP4/anger/F02-Anger-Face Forward-11.mp4"; // Anger
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/ADFES_Labelled_MP4/anger/F03-Anger-Face Forward-21.mp4"; // Sad

		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/ADFES_Labelled_MP4/fear/M01-Fear-Face Forward-52.mp4"; // Fear
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/ADFES_Labelled_MP4/fear/M02-Fear-Face Forward-55.mp4"; // Fear
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/ADFES_Labelled_MP4/fear/M03-Fear-Face Forward-65.mp4"; // Fear
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/ADFES_Labelled_MP4/fear/F01-Fear-Face Forward-5.mp4"; // Fear
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/ADFES_Labelled_MP4/fear/F02-Fear-Face Forward-15.mp4"; // Neutral
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/ADFES_Labelled_MP4/fear/F03-Fear-Face Forward-25.mp4"; // Fear

		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/ADFES_Labelled_MP4/disgust/M01-Disgust-Face Forward-50.mp4"; // Surprise
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/ADFES_Labelled_MP4/disgust/M02-Disgust-Face Forward-53.mp4"; // Disgust
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/ADFES_Labelled_MP4/disgust/M03-Disgust-Face Forward-63.mp4"; // Anger
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/ADFES_Labelled_MP4/disgust/F01-Disgust-Face Forward-3.mp4"; // Surprise
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/ADFES_Labelled_MP4/disgust/F02-Disgust-Face Forward-13.mp4"; // Neutral
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/ADFES_Labelled_MP4/disgust/F03-Disgust-Face Forward-23.mp4"; // Neutral
		
		
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/ADFES_Labelled_MP4/surprise/M01-Surprise-Face Forward-57.mp4"; // Surprise
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/ADFES_Labelled_MP4/surprise/M02-Surprise-Face Forward-60.mp4"; // Surprise
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/ADFES_Labelled_MP4/surprise/M03-Surprise-Face Forward-70.mp4"; // Surprise
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/ADFES_Labelled_MP4/surprise/F01-Surprise-Face Forward-10.mp4"; // Surprise
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/ADFES_Labelled_MP4/surprise/F02-Surprise-Face Forward-20.mp4"; // Surprise
		//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/ADFES_Labelled_MP4/surprise/F03-Surprise-Face Forward-30.mp4"; // Surprise
		
		
		// ==============================================================================================================================

		int _width = 640;
		int _height = 480;		
		Mat rgbVideoMat;
		Texture2D Video_texture;		
		
		static readonly string CSV_EXPORT = "unity_au_values.csv";
		//StreamWriter csv_writer;
		
		VideoWriter videoWriter;	
		Mat recordingFrameRgbMat;
		
		// Start is called before the first frame update
		private IEnumerator Start()
		{	
			
			if (targetUMA != null)
			{ 
				expressionPlayer = targetUMA.GetComponent<ExpressionPlayer>();
			}
			InitialiseAvatarSettings();
			overallFacialInput = new List<structAU>();
			facialInput = new List<structAU>();
			
			
			// ============================================================================
			// Setup Video Output File
			Debug.Log("==============================");
			//Debug.Log("Camera: " + MainCam.pixelWidth + "/" + MainCam.pixelHeight);	
			videoWriter = new VideoWriter();
			videoWriter.open(Application.persistentDataPath + "/output.mp4", VideoWriter.fourcc('X', '2', '6', '4'), 30, new Size((int)MainCam.pixelWidth, (int)MainCam.pixelHeight));
			recordingFrameRgbMat = new Mat((int)MainCam.pixelHeight, (int)MainCam.pixelWidth, CvType.CV_8UC3);
			Debug.Log("File : " + Application.persistentDataPath + "/output.mp4");	

			if (!videoWriter.isOpened())
			{
				Debug.LogError("videoWriter.isOpened() false");
				//videoWriter.release();
			}			

			// ============================================================================
			// Load ML models			
			m_ImageRuntimeModel = ModelLoader.Load(image_model);
			m_AudioRuntimeModel = ModelLoader.Load(audio_model);

			image_worker = WorkerFactory.CreateWorker(WorkerFactory.Type.CSharpRef, m_ImageRuntimeModel); 
			audio_worker = WorkerFactory.CreateWorker(WorkerFactory.Type.CSharpRef, m_AudioRuntimeModel); 

			// ============================================================================
			// Set variables			
			
			audioSource = GetComponent<AudioSource>();
			audio_length = Frequency * recognitionIntervalTime;
			Debug.Log("audio_length : " + audio_length);	
			Start_Recording = true;
			Recording_FirstTime = true;	
			//recognitionTime = recognitionIntervalTime;
			recognitionFrames = recognitionIntervalFrames;			
			ResetBufferTotals();

			// ============================================================================
			
			_screen.rectTransform.sizeDelta = new Vector2(_width, _height);
			_inputTexture = new Texture2D(_width, _height, TextureFormat.RGBA32, false);
			_inputPixelData = new Color32[_width * _height];
			_outputTexture = new Texture2D(_width, _height, TextureFormat.RGBA32, false);
			_outputPixelData = new Color32[_width * _height];	  

			_screen.texture = _outputTexture;

			_resourceManager = new LocalResourceManager();
			yield return _resourceManager.PrepareAssetAsync("face_detection_short_range.bytes");
			yield return _resourceManager.PrepareAssetAsync("face_landmark_with_attention.bytes");

			stopwatch = new Stopwatch();

			_graph = new CalculatorGraph(_configAsset.text);
			_outputVideoStream = new OutputStream<ImageFrame>(_graph, "output_video");
			_multiFaceLandmarksStream = new OutputStream<List<NormalizedLandmarkList>>(_graph, "multi_face_landmarks");
			_outputVideoStream.StartPolling();
			_multiFaceLandmarksStream.StartPolling();
			_graph.StartRun();		
			stopwatch.Start();	  
		  
			screenRect = _screen.GetComponent<RectTransform>().rect;	
			Debug.Log("==============================");
			Debug.Log(screenRect);
			//Debug.Log($"Unity Local Coordinates: {screenRect.GetPoint(test)}, Image Coordinates: {test}");
			Debug.Log("* " + screenRect.GetType().ToString());
			Debug.Log("==============================");
			
			while (true)
			{

				ProcessVideo();
				// Start_Recording

				if (currentFrame >= previousFrame)
				{
					
					var imageFrame = new ImageFrame(ImageFormat.Types.Format.Srgba, _width, _height, _width * 4, _inputTexture.GetRawTextureData<byte>());
					var currentTimestamp = stopwatch.ElapsedTicks / (System.TimeSpan.TicksPerMillisecond / 1000);
					_graph.AddPacketToInputStream("input_video", Packet.CreateImageFrameAt(imageFrame, currentTimestamp));

					var task1 = _outputVideoStream.WaitNextAsync();
					var task2 = _multiFaceLandmarksStream.WaitNextAsync();
					var task = Task.WhenAll(task1, task2);

					yield return new WaitUntil(() => task.IsCompleted);

					if (!task1.Result.ok || !task2.Result.ok)
					{
						throw new System.Exception("Something went wrong");
					}

					var outputVideoPacket = task1.Result.packet;
					if (outputVideoPacket != null)
					{
						var outputVideo = outputVideoPacket.Get();
						if (outputVideo.TryReadPixelData(_outputPixelData))
						{
							//_outputTexture.SetPixels32(_outputPixelData);
							//_outputTexture.Apply();
							
							
							
						}
					}

					var multiFaceLandmarksPacket = task2.Result.packet; 
					if (multiFaceLandmarksPacket != null)
					{
						var multiFaceLandmarks = multiFaceLandmarksPacket.Get(NormalizedLandmarkList.Parser);
						if (multiFaceLandmarks != null && multiFaceLandmarks.Count > 0)
						{
							
							//ProcessFace(multiFaceLandmarks);
							
							foreach (var landmarks in multiFaceLandmarks)
							{
								
								ProcessFace(landmarks);
								
								// top of the head
								//var topOfHead = landmarks.Landmark[10];
								//Debug.Log($"Unity Local Coordinates: {screenRect.GetPoint(topOfHead)}, Image Coordinates: {topOfHead}");
								
								// Draw onto _inputTexture then apply it as _outputTexture
								//_outputTexture.SetPixels(_inputTexture.GetPixels());
								//_outputTexture.Apply();
								
							}
						}
					}
					
				}
				
				previousFrame = currentFrame;
				
				UpdateScreen();	
				
				
				// Write Video Frame
				Utils.texture2DToMat(GetTextureFromCamera(MainCam), recordingFrameRgbMat);
				Imgproc.cvtColor(recordingFrameRgbMat, recordingFrameRgbMat, Imgproc.COLOR_RGB2BGR);
				videoWriter.write(recordingFrameRgbMat);

				
			}

			
		}
		
        private static Texture2D GetTextureFromCamera(Camera mCamera)
        {
            UnityEngine.Rect rect = new UnityEngine.Rect(0, 0, mCamera.pixelWidth, mCamera.pixelHeight);
            RenderTexture renderTexture = new RenderTexture(mCamera.pixelWidth, mCamera.pixelHeight, 24);
            Texture2D screenShot = new Texture2D(mCamera.pixelWidth, mCamera.pixelHeight, TextureFormat.RGBA32, false);
     
            mCamera.targetTexture = renderTexture;
            mCamera.Render();
     
            RenderTexture.active = renderTexture;
     
            screenShot.ReadPixels(rect, 0, 0);
            screenShot.Apply();
     
     
            mCamera.targetTexture = null;
            RenderTexture.active = null;
            return screenShot;
        }
		
		
		void ProcessVideo ()	
		{
			int temp;
			//Debug.Log("Process video:" + counter);
			
			// Initialise video
			if (!video_capture_initialised) 
			{

				currentFrame = 1;
				previousFrame = 0;
		
				Debug.Log("Initialise video");
				rgbVideoMat = new Mat();

				capture = new VideoCapture();
				capture.open(OpenCVForUnity.UnityUtils.Utils.getFilePath(VIDEO_FILENAME));
				video_capture_initialised = true;
				
				capture.grab();
				capture.retrieve(rgbVideoMat);
				Video_texture = new Texture2D(rgbVideoMat.cols(), rgbVideoMat.rows(), TextureFormat.RGB24, false);
				
				
			} else {

				//shouldUpdateVideoFrame = false;
			
				// Loop play
				if ((capture.get(Videoio.CAP_PROP_POS_FRAMES) + 1) >= (capture.get(Videoio.CAP_PROP_FRAME_COUNT) - 5))
				{
					capture.set(Videoio.CAP_PROP_POS_FRAMES, 0);
					Debug.Log("=============================");
					Debug.Log("Video completed");
					Debug.Log("============================");
					
					// ===========================================================================
					// Write CSV containing action unit values
					StreamWriter writer = new StreamWriter(Application.persistentDataPath + "/" + CSV_EXPORT);
					writer.WriteLine("AU1,AU2,AU4,AU6,AU7,AU9,AU15,AU20,AU23,AU25,AU26");
					for (var x = 0; x < overallFacialInput.Count; x++)
					{
						writer.WriteLine(overallFacialInput[x].AU1 + "," + overallFacialInput[x].AU2 + "," + overallFacialInput[x].AU4 + "," + overallFacialInput[x].AU6 + "," + overallFacialInput[x].AU7 + "," + overallFacialInput[x].AU9 + "," + overallFacialInput[x].AU15 + "," + overallFacialInput[x].AU20 + "," + overallFacialInput[x].AU23 + "," + overallFacialInput[x].AU25 + "," + overallFacialInput[x].AU26);
					}
					writer.Flush();
					writer.Close();
					
					videoWriter.release();	

				
					// ==================================

			
					
					Debug.Log("bufferTotals:" + bufferTotals[0] + "/" + bufferTotals[1] + "/" + bufferTotals[2] + "/" + bufferTotals[3] + "/" + bufferTotals[4] + "/" + bufferTotals[5] + "/" + bufferTotals[6]);
					var maxValue = bufferTotals.Max();
					var maxIndex = bufferTotals.ToList().IndexOf(maxValue);
					
					Debug.Log("Frames=" + currentFrame + " Max:" + observed_emotions[maxIndex] + "(" + maxValue + ")");
					

					// ============================================================================
					// ============================================================================
					// Obtain Overall Facial Prediction
					imageBuffer = facialEmotionPrediction(overallFacialInput);
					ResetBufferTotals();					
					for(int emotion = 0; emotion < numEmotions; emotion++)
					{
						row_average = 0f;
						for(int row = 0; row < imageBuffer.Count; row++) { row_average += imageBuffer[row][emotion]; }
						bufferTotals[emotion] = (float)(Math.Round(row_average / imageBuffer.Count,2)); 
					}
					
					maxValue = bufferTotals.Max();
					maxIndex = bufferTotals.ToList().IndexOf(maxValue);
					Debug.Log("Overall imageBuffer:" + (int)(bufferTotals[0]) + "/" + (int)(bufferTotals[1]) + "/" + (int)(bufferTotals[2]) + "/" + (int)(bufferTotals[3]) + "/" + (int)(bufferTotals[4]) + "/" + (int)(bufferTotals[5]) + "/" + (int)(bufferTotals[6]));
					Debug.Log("Overall facial prediction=" + currentFrame + " Max:" + observed_emotions[maxIndex] + "(" + maxValue + ")");
					
					
					
					// ============================================================================
					// Analyse all audio
					float[] clipData = new float[audioSource.clip.samples * audioSource.clip.channels];					
					audioBuffer = audioEmotionPrediction(clipData, 0);
					maxValue = audioBuffer[0].Max();
					maxIndex = audioBuffer[0].ToList().IndexOf(maxValue);
					
					Debug.Log("Overall audioBuffer:" + (int)(audioBuffer[0][0]) + "/" + (int)(audioBuffer[0][1]) + "/" + (int)(audioBuffer[0][2]) + "/" + (int)(audioBuffer[0][3]) + "/" + (int)(audioBuffer[0][4]) + "/" + (int)(audioBuffer[0][5]) + "/" + (int)(audioBuffer[0][6]));
					
					Debug.Log("Overall audio prediction=" + currentFrame + " Max:" + observed_emotions[maxIndex] + "(" + maxValue + ")");

					
					// ============================================================================
					// Overall Combined
					for(int emotion = 0; emotion < numEmotions; emotion++)
					{
						bufferTotals[emotion] += (float)(audioBuffer[0][emotion]); 
					}
					maxValue = bufferTotals.Max();
					maxIndex = bufferTotals.ToList().IndexOf(maxValue);
					Debug.Log("Overall prediction=" + currentFrame + " Max:" + observed_emotions[maxIndex] + "(" + maxValue + ")");




					
					// ============================================================================
					
					
					
					overallFacialInput.Clear();				
					facialInput.Clear();
					bufferTotals[0] = 0f;
					bufferTotals[1] = 0f;
					bufferTotals[2] = 0f;
					bufferTotals[3] = 0f;
					bufferTotals[4] = 0f;
					bufferTotals[5] = 0f;
					bufferTotals[6] = 0f;
				
					
				}

				if (capture.grab())
				{

					capture.retrieve(rgbVideoMat);

				}

			}	

			//Debug.Log("Capture video frame" + capture.get(Videoio.CAP_PROP_POS_FRAMES) + " / " + capture.get(Videoio.CAP_PROP_FRAME_COUNT) );				
			
			currentFrame = (int)capture.get(Videoio.CAP_PROP_POS_FRAMES);
			//Debug.Log("currentFrame=" + currentFrame + "   previousFrame=" + previousFrame);
			Texture2D tex2D = new Texture2D(_width, _height, TextureFormat.RGBA32, false);
			Imgproc.cvtColor(rgbVideoMat, rgbVideoMat, Imgproc.COLOR_BGR2RGB);
			
			OpenCVForUnity.UnityUtils.Utils.fastMatToTexture2D(rgbVideoMat, Video_texture);
			tex2D = Video_texture;
			
			_inputTexture.SetPixels32(tex2D.GetPixels32());
			shouldUpdateVideoFrame = false;	


		}	
		

		// Update is called once per frame
		void UpdateScreen()
		{
			var last_index = 0;

			if (Start_Recording) {
				
				if (Recording_FirstTime) {

					Recording_FirstTime = false;
					Debug.Log("START RECORDING");
				
					//set up recording to last a max of 1 seconds and loop over and over
					//audioSource.clip = Microphone.Start(null, true, 1, Frequency);
					//audioSource.Play();

				}


				//if (recognitionTime > 0)
				if (recognitionFrames > 0)
				{
					
					//recognitionTime -= Time.deltaTime;	//Subtract elapsed time every frame
					//Debug.Log("recognitionTime: " + recognitionTime + "  Time.deltaTime=" + Time.deltaTime);
					recognitionFrames -= 1;
					
				} else {
					
					//Debug.Log("Image Count: " + facialInput.Count.ToString() + "  First AU1=" + facialInput[0].AU1.ToString());

					// ============================================================================
					// Obtain Facial Prediction - convert 1D array into 2D array 
					imageBuffer = facialEmotionPrediction(facialInput);					
					//Debug.Log("imageBuffer Size:" + imageBuffer.Count + "/" + imageBuffer[0].Length);
					//Debug.Log("imageBuffer[0]" + imageBuffer[0][0] + "-" + imageBuffer[0][6]);								
			
					
					// ============================================================================
					// Extract required audio
					//add the next second(s) of recorded audio to temp vector
					var channelCount = audioSource.clip.channels;
					Debug.Log("audio values:" + audio_length + "-" + channelCount + "-" + recognitionIntervalTime + " - " + Frequency);
					int clipSize = (int)((audio_length / channelCount) * recognitionIntervalTime);
					//Debug.Log("clipSize:" + clipSize);
					float[] clipData = new float[clipSize];
					
					audioBuffer = audioEmotionPrediction(clipData, audioBytesRead);
					audioBytesRead += clipSize;
					
					//Debug.Log("currentFrame=" + currentFrame + " audioBuffer:" + audioBuffer[0][0] + "/" + audioBuffer[0][1] + "/" + audioBuffer[0][2] + "/" + audioBuffer[0][3] + "/" + audioBuffer[0][4] + "/" + audioBuffer[0][5] + "/" + audioBuffer[0][6]);					
					
					
					
					// ============================================================================
					// Reset Timer
					//recognitionTime = recognitionIntervalTime;	
					recognitionFrames = recognitionIntervalFrames;	
					
					
					// ============================================================================
					// Accumulate the image and audio predictions
					
					Debug.Log("*Size:" + facialInput.Count.ToString() + "/" + imageBuffer.Count + "/" + audioBuffer.Count);
					for(int emotion = 0; emotion < numEmotions; emotion++)
					{
						fusedBuffer[emotion] = 0f;
						row_average = 0f;
						for(int row = 0; row < imageBuffer.Count; row++)
						{
							row_average += imageBuffer[row][emotion];
							//Debug.Log("row_average=" + row_average);
						}
						row_average = (row_average / imageBuffer.Count);	
						if (audioBuffer.Count < 1) {
							// Image only
							fusedBuffer[emotion] = (float)Math.Round(row_average,2);
						} else {
							// Average of Image and Audio
							
							fusedBuffer[emotion] = (float)Math.Round(row_average,2);						
							fusedBuffer[emotion] += (float)Math.Round(audioBuffer[0][emotion],2);
							
							//fusedBuffer[emotion] = (float)Math.Round((((row_average + audioBuffer[0][emotion]) / 2) * 100),2);

						}
						bufferTotals[emotion] += (float)(fusedBuffer[emotion]);  //(int)(fusedBuffer[emotion] * 100);
					}

					Debug.Log("currentFrame=" + currentFrame + " audioBuffer:" + audioBuffer[0][0] + "/" + audioBuffer[0][1] + "/" + audioBuffer[0][2] + "/" + audioBuffer[0][3] + "/" + audioBuffer[0][4] + "/" + audioBuffer[0][5] + "/" + audioBuffer[0][6]);
					
					Debug.Log("currentFrame=" + currentFrame + " imageBuffer:" + imageBuffer[0][0] + "/" + imageBuffer[0][1] + "/" + imageBuffer[0][2] + "/" + imageBuffer[0][3] + "/" + imageBuffer[0][4] + "/" + imageBuffer[0][5] + "/" + imageBuffer[0][6]);


					Debug.Log("currentFrame=" + currentFrame + " FusedBuffer:" + fusedBuffer[0] + "/" + fusedBuffer[1] + "/" + fusedBuffer[2] + "/" + fusedBuffer[3] + "/" + fusedBuffer[4] + "/" + fusedBuffer[5] + "/" + fusedBuffer[6]);
					
					Debug.Log("currentFrame=" + currentFrame + " bufferTotals:" + bufferTotals[0] + "/" + bufferTotals[1] + "/" + bufferTotals[2] + "/" + bufferTotals[3] + "/" + bufferTotals[4] + "/" + bufferTotals[5] + "/" + bufferTotals[6]);
					
					Progress_Neutral.BarValue += fusedBuffer[0];
					Progress_Happy.BarValue += fusedBuffer[1];
					Progress_Sad.BarValue += fusedBuffer[2];
					Progress_Anger.BarValue += fusedBuffer[3];
					Progress_Fear.BarValue += fusedBuffer[4];
					Progress_Disgust.BarValue += fusedBuffer[5];
					Progress_Surprise.BarValue += fusedBuffer[6];

					//Progress_Neutral.BarValue += bufferTotals[0];
					//Progress_Happy.BarValue += bufferTotals[1];
					//Progress_Sad.BarValue += bufferTotals[2];
					//Progress_Anger.BarValue += bufferTotals[3];
					//Progress_Fear.BarValue += bufferTotals[4];
					//Progress_Disgust.BarValue += bufferTotals[5];
					//Progress_Surprise.BarValue += bufferTotals[6];
					
					// ============================================================================
					// Clear inputs
					facialInput.Clear();
					//image_output.Dispose();					
					//audio_output.Dispose();
					// ============================================================================

				}

			}
			
			// Update AU onscreen indicators
			if (facialInput.Count < 1) {
				
				Debug.Log("ERROR: No facial input");
				
			} else {
				
				last_index = facialInput.Count - 1;
				Display_AU1.text = "AU1 = " + facialInput[last_index].AU1.ToString();
				Display_AU2.text = "AU2 = " + facialInput[last_index].AU2.ToString();
				Display_AU4.text = "AU4 = " + facialInput[last_index].AU4.ToString();
				Display_AU6.text = "AU6 = " + facialInput[last_index].AU6.ToString();
				Display_AU7.text = "AU7 = " + facialInput[last_index].AU7.ToString();
				Display_AU9.text = "AU9 = " + facialInput[last_index].AU9.ToString();
				Display_AU15.text = "AU15 = " + facialInput[last_index].AU15.ToString();
				Display_AU20.text = "AU20 = " + facialInput[last_index].AU20.ToString();
				Display_AU23.text = "AU23 = " + facialInput[last_index].AU23.ToString();
				Display_AU25.text = "AU25 = " + facialInput[last_index].AU25.ToString();
				Display_AU26.text = "AU26 = " + facialInput[last_index].AU26.ToString();
				//Debug.Log("AU's displayed");

				Progress_Neutral.BarValue -= ADSR_release;
				Progress_Happy.BarValue -= ADSR_release;
				Progress_Sad.BarValue -= ADSR_release;
				Progress_Anger.BarValue -= ADSR_release;
				Progress_Fear.BarValue -= ADSR_release;
				Progress_Disgust.BarValue -= ADSR_release;
				Progress_Surprise.BarValue -= ADSR_release;
				
			}

			
		}	
		
		private List<float[]> facialEmotionPrediction(List<structAU> fInput)
		{
			float[] outputBuffer;
			List<float[]> rtn_buffer = new List<float[]>();
			//float[] predictionArray = new float[numEmotions];
			//int i = 0;
			int j = 0;
		
			// ============================================================================
			// Create Facial Tensor and perform Facial ML prediction
			Tensor facial_input = new Tensor(fInput.Count, 1, 1, 11); 				
			for (var x = 0; x < fInput.Count; x++)
			{
				facial_input[x,0,0,0] = fInput[x].AU1;
				facial_input[x,0,0,1] = fInput[x].AU2;
				facial_input[x,0,0,2] = fInput[x].AU4;
				facial_input[x,0,0,3] = fInput[x].AU6;
				facial_input[x,0,0,4] = fInput[x].AU7;
				facial_input[x,0,0,5] = fInput[x].AU9;
				facial_input[x,0,0,6] = fInput[x].AU15;
				facial_input[x,0,0,7] = fInput[x].AU20;
				facial_input[x,0,0,8] = fInput[x].AU23;
				facial_input[x,0,0,9] = fInput[x].AU25;
				facial_input[x,0,0,10] = fInput[x].AU26;

			}
			image_worker.Execute(facial_input);

			Tensor image_output = image_worker.PeekOutput("modelOutput");	
			outputBuffer = image_output.ToReadOnlyArray();

			// =======================================================================
			// Convert 1D OutputBuffer to a List of 1D Predictions (easier to manage)
			for(int i = 0; i < outputBuffer.Length; i+=numEmotions)
			{
				float[] predictionArray = new float[numEmotions];
				predictionArray[0] = (float) outputBuffer[i] * 100;
				predictionArray[1] = (float) outputBuffer[i+1] * 100;
				predictionArray[2] = (float) outputBuffer[i+2] * 100;
				predictionArray[3] = (float) outputBuffer[i+3] * 100;
				predictionArray[4] = (float) outputBuffer[i+4] * 100;
				predictionArray[5] = (float) outputBuffer[i+5] * 100;
				predictionArray[6] = (float) outputBuffer[i+6] * 100;
				
				rtn_buffer.Add(predictionArray);

			}
			
			
			return rtn_buffer;
		}

		private List<float[]> audioEmotionPrediction(float[] clipData, int start_point)
		{
			float mean_value = 0;
			float[] outputBuffer;
			List<float[]> rtn_buffer = new List<float[]>();
			float[] predictionArray = new float[numEmotions];			


			float[] samples = new float[audioSource.clip.samples * audioSource.clip.channels];
			audioSource.clip.GetData(samples, 0);
			//Debug.Log("GetData (1): " + clipData.Length.ToString() + " " + samples.Length.ToString());
			 
			for (int i = 0; i < clipData.Length; ++i)
			{
				if ((i + start_point) <= samples.Length) { clipData[i] = samples[i + start_point]; }
			}
			
			// ============================================================================
			// Analyse audio to extract MFCC and create Audio Tensor - based on Cepstrums
			Tensor audio_input = new Tensor(1, 1, 1, Cepstrum_length-1); 
			int sampleRate = audioSource.clip.frequency;
			//DiscreteSignal signal = Signal.FromArray(clipData, sampleRate);
			DiscreteSignal signal = new DiscreteSignal(sampleRate, clipData);
		
		
		
			// ============================================================================
		
			var melBank1 = FilterBanks.MelBankSlaney(filterbankSize, fftSize, sampleRate, lowFreq, highFreq);
			var opts = new MfccOptions
			{
				SamplingRate = sampleRate,
				FrameDuration = (double)fftSize / sampleRate,
				HopDuration = 0.010,
				FeatureCount = Cepstrum_length,
				FilterBank = melBank1,  					
				NonLinearity = NonLinearityType.ToDecibel, 	
				Window = WindowType.Hamming,     			
				LogFloor = 1e-10f,  						
				DctType="2N",
				LifterSize = 0
			};
			var extractor = new MfccExtractor(opts);
			var mfccVectors = extractor.ComputeFrom(signal);
			//Debug.Log("mfccVectors count: " + mfccVectors.Count.ToString() + " - " + mfccVectors[0].Length.ToString());			

			// ============================================================================
			// Pre-process MFCC - find the mean cell value, then subtract the mean from each cell.
			for(int column = 1; column < Cepstrum_length; column++)
			{
				for(int row = 0; row < mfccVectors.Count; row++) { mean_value += (float)mfccVectors[row][column]; }					
				mean_value /= ((Cepstrum_length - 1) * mfccVectors.Count);		
			}					

			for(int column = 1; column < Cepstrum_length; column++)
			{
				for(int row = 0; row < mfccVectors.Count; row++) { mfccVectors[row][column] -= mean_value; }					
			}								
			
			// ============================================================================
			// Fill Audio Tensor with averages across Cepstrums (but leave out first one!!!!!!!) - not for now
			for(int column = 1; column < Cepstrum_length; column++)
			{
				row_average = 0f;
				for(int row = 0; row < mfccVectors.Count; row++) { row_average += (float)mfccVectors[row][column]; }					
				row_average /= mfccVectors.Count;		
				audio_input[0,0,0,column-1] = row_average;	
			}					

			
			// ============================================================================
			// Perform Audio ML prediction
			audio_worker.Execute(audio_input);
			
			Tensor audio_output = audio_worker.PeekOutput("modelOutput");	
			outputBuffer = audio_output.ToReadOnlyArray();				
			//Debug.Log("Audio:" + audioBuffer[0] + " , " + audioBuffer[1] + " , " + audioBuffer[2] + " , " + audioBuffer[3] + " , " + audioBuffer[4] + " , " + audioBuffer[5] + " , " + audioBuffer[6]); //print the result as a percentage
			
			// =======================================================================
			// Convert 1D OutputBuffer to a List of 1D Predictions (easier to manage)
			for(int i = 0; i < outputBuffer.Length; i+=numEmotions)
			{
				for(int j = 0; j < numEmotions; j++)
				{
					predictionArray[j] = (float)outputBuffer[i+j] * 100;
				}
				rtn_buffer.Add(predictionArray);
			}
			
			return rtn_buffer;
		}
		
		private void ResetBufferTotals()
		{
		
			for(int emotion = 0; emotion < numEmotions; emotion++)
			{
				bufferTotals[emotion] = 0f;
			}		

		}
		
		public void SetupScreenRect(int rect_width, int rect_height)
		{

			_screen.rectTransform.sizeDelta = new Vector2(rect_width, rect_height);					
			screenRect = _screen.GetComponent<RectTransform>().rect;
			screenRect.x = 0;			
			screenRect.y = 0;			
			
			//Debug.Log("==============================");
			//Debug.Log(screenRect);
		
		}
		
		public void ProcessFace(Mediapipe.NormalizedLandmarkList mp_normalisedlandmarks)	
        {
			
			//var topOfHead = mp_normalisedlandmarks.Landmark[10];
			//Debug.Log($"Unity Local Coordinates: {screenRect.GetPoint(topOfHead)}, Image Coordinates: {topOfHead}");
			
			List<Vector2> points;
			//HighCol = new Color(200,200,200); 
			
			
			/*
			points = new List<Vector2>();
			points = ConvertMediaPipeLandmarksToVectorList(mp_normalisedlandmarks);
			if ( points.Count == 68) { 
				bool success = CalculateAvatarSettings (points);
				if ( success ) { 
					overallFacialInput.Add(faceAU);
					facialInput.Add(faceAU);
				}	

				_outputTexture.SetPixels(_inputTexture.GetPixels());			
				for (int i = 0; i < numFinalLandmarkPoints; i++)
				{
					_outputTexture.SetPixel( (int)points[i].x, (int)(_height - points[i].y), UnityEngine.Color.white );
				}					
				_outputTexture.Apply();

				
			} else {
				Debug.Log("Points <> 68");
			}
			*/
			
			
			points = new List<Vector2>();
			points = ConvertMediaPipeLandmarksToVectorList(mp_normalisedlandmarks);
			if ( points.Count == numFaceMeshLandmarkPoints) { 
				bool success = CalculateAvatarSettings (points);
				if ( success ) { 
					overallFacialInput.Add(faceAU);
					facialInput.Add(faceAU);
				}	

				_outputTexture.SetPixels(_inputTexture.GetPixels());			
				for (int i = 0; i < numFaceMeshLandmarkPoints; i++)
				{
					_outputTexture.SetPixel( (int)points[i].x, (int)(_height - points[i].y), UnityEngine.Color.white );
				}					
				_outputTexture.Apply();

				
			} else {
				Debug.Log("Points <> 68");
			}
						
			
		}	

        private List<Vector2> ConvertMediaPipeLandmarksToVectorList(Mediapipe.NormalizedLandmarkList mp_pts, List<Vector2> pts = null)
        {
			/*
			// Jawline 0-15
			// Right-eyebrow 16-21
			// Left-eyebrow 22-26
			// Nose 27-35
			// Eyes 36-47
			// Mouth 48-68
			int[] original_landmarks_68 = new int[] {162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,
								70,63,105,66,193,
								417,296,334,293,300,
								168,197,5,4,75,97,2,326,305,
								33,160,158,133,153,144,362,385,387,263,373,380,
			61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87};		
		
			int[] landmarks_68 = new int[] {162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,
									70,63,105,66,55,
									285,296,334,293,300,
									168,197,5,4,75,97,2,326,305,
									33,160,158,133,153,144,362,385,387,263,373,380,
									61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87};

			int[] alt_landmarks_68 = new int[] {162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,
									71,63,105,66,107,336,
									296,334,293,301,168,197,5,4,75,97,2,326,305,33,160,158,133,153,144,362,385,387,263,373,
									380,61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87};

			*/
			
			Vector3 temp; 
			
            if (pts == null)
            {
                pts = new List<Vector2>();
            }
            pts.Clear();

			/*
			for (int i = 0; i < numFinalLandmarkPoints; i++)
			{
				//float x = (float)Math.Round(((mp_pts.Landmark[landmarks_68[i]].X) * _width),2);
				//float y = (float)Math.Round(((mp_pts.Landmark[landmarks_68[i]].Y) * _height),2);
				int x = (int)(mp_pts.Landmark[landmarks_68[i]].X * _width);
				int y = (int)(mp_pts.Landmark[landmarks_68[i]].Y * _height);
				pts.Add(new Vector2(x, y));
				
			}
			*/

			for (int i = 0; i < numFaceMeshLandmarkPoints; i++)
			{
				int x = (int)(mp_pts.Landmark[i].X * _width);
				int y = (int)(mp_pts.Landmark[i].Y * _height);
				pts.Add(new Vector2(x, y));
				
			}
			
            return pts;
        }
		
        protected void InitialiseAvatarSettings ()
        {
			
			if (refine_AU_extents) {

				AU1_low = 0.4f; AU1_high = 2f;

				AU2_l_low = 1.3f; AU2_l_high = 2f;  			
				AU2_r_low = 1.3f; AU2_r_high = 2f; 		

				AU4_low = 2.5f; AU4_high = 6f;
				
				AU6_low = 1.7f; AU6_high = 2.5f; 				

				AU7_l_low = 0.4f; AU7_l_high = 0.001f; 
				AU7_r_low = 0.4f; AU7_r_high = 0.001f;
					
				AU9_low = 3f; AU9_high = 0.5f;

				AU15_l_low = 2.25f; AU15_l_high = -1f;
				AU15_r_low = 2.25f; AU15_r_high = -1f;
				
				AU20_low = 6f; AU20_high = 0.4f;				
				AU23_low = 8f; AU23_high = 0.1f;
				AU26_low = 2f; AU26_high = 0.0001f;			
			
				/*
				AU1_low = 3f; AU1_high = 0.05f;
				AU2_l_low = 3f; AU2_l_high = 0.9f;  			
				AU2_r_low = 3f;	AU2_r_high = 0.9f; 		
				AU4_low = 8f; AU4_high = 1f;
				AU6_low = 3f; AU6_high = 1f; 				
				//whAU6_l_low = 1.5f; AU6_l_high = 0.5f; 				
				//AU6_r_low = 1.5f; AU6_r_high = 0.5f;                                                                       
				AU7_l_low = 0.4f; AU7_l_high = 0.001f; 
				AU7_r_low = 0.4f; AU7_r_high = 0.001f;
				AU9_low = 3f; AU9_high = 0.5f;
				AU15_l_low = 2.25f; AU15_l_high = -1f;
				AU15_r_low = 2.25f; AU15_r_high = -1f;
				AU20_low = 6f; AU20_high = 0.4f;				
				AU23_low = 8f; AU23_high = 0.1f;
				AU26_low = 2f; AU26_high = 0.0001f; 		
				*/
				
			} else {
				
				AU1_low = 0.1f; AU1_high = 2.8f;

				AU2_l_low = 0.8f; AU2_l_high = 3f; 			
				AU2_r_low = 0.8f; AU2_r_high = 3f;		
			
				AU4_low = 2f; AU4_high = 7f; 
							
				AU6_low = 1.5f; AU6_high = 2.8f; 				
				//AU6_l_low = 0.75f; AU6_l_high = 1.4f;  
				//AU6_r_low = 0.75f; AU6_r_high = 1.4f;                                                                                                        

				AU7_l_low = 0.01f; AU7_l_high = 0.5f; 				 
				AU7_r_low = 0.01f; AU7_r_high = 0.5f;
					
				AU9_low = 1.1f; AU9_high = 2.7f;

				AU15_l_low = -1.1f; AU15_l_high = 2.25f; 
				AU15_r_low = -1.1f; AU15_r_high = 2.25f;
				
				AU20_low = 0.4f; AU20_high = 6f;	
				
				AU23_low = 2.1f; AU23_high = 9.5f; 
				AU26_low = 0f; AU26_high = 1f; 		
				
			}
			
		}

        protected virtual bool CalculateAvatarSettings (List<Vector2> points)
        {
			
		    //double widthOfLeftEyeSocket;
            //double widthOfRightEyeSocket;
            //double widthOfEyeSockets;
			//double LengthOfNose;

			//double widthOfLeftSideOfFace;
            //double widthOfRightSideOfFace;
            //double widthOfFace;

            //double heightOfLeftSideOfFace;
            //double heightOfRightSideOfFace;
            //double heightOfUpperFace;
            //double heightOfLowerFace;
            //double distanceOfLeftCheek;
            //double distanceOfRightCheek;

            float Temp;
            float Temp2;

			float AU1 = 0; float AU1_r = 0; float AU1_l;
			float AU2 = 0; float AU2_r = 0; float AU2_l;
			float AU4 = 0; float AU4_r = 0; float AU4_l;
			float AU6 = 0; float AU6_r = 0; float AU6_l;
			float AU7 = 0; float AU7_r = 0; float AU7_l;
			float AU9 = 0;
			float AU15 = 0; float AU15_r = 0; float AU15_l;
			float AU20 = 0;
			float AU23 = 0;
			float AU25 = 0;
			float AU26 = 0;
			
			float AU1_lowPassFilter = 1f;
			float AU2_lowPassFilter = 1f;
			float AU4_lowPassFilter = 1f;	
			float AU6_lowPassFilter = 1f;	
			float AU7_lowPassFilter = 1f;
			float AU9_lowPassFilter = 1f;	
			float AU15_lowPassFilter = 1f;
			float AU20_lowPassFilter = 1f;			
			float AU23_lowPassFilter = 0.5f;
			float AU26_lowPassFilter = 0.75f;
			bool rtn = false; 
		
			if (targetUMA != null)
			{

				//expressionPlayer = targetUMA.GetComponent<ExpressionPlayer>();
				if (expressionPlayer == null)
				{
					Debug.Log("No expression player!");
					return rtn;
				}
				else
				{

					// =====================================================================
					// Work out expressions
					// =====================================================================

					rtn = true;
					counter++;
					//widthOfLeftEyeSocket = distance(points[27].x,points[27].y,points[36].x,points[36].y);
					//widthOfRightEyeSocket = distance(points[27].x,points[27].y,points[45].x,points[45].y);
					//widthOfEyeSockets = widthOfLeftEyeSocket + widthOfRightEyeSocket;
					//LengthOfNose = distance(points[27].x,points[27].y,points[30].x,points[30].y);

					//heightOfLeftSideOfFace = distance(0,points[0].y,0,points[8].y);
					//heightOfRightSideOfFace = distance(0,points[16].y,0,points[8].y);				

					//widthOfLeftSideOfFace = new Vector2(points[28].x - points[0].x, 1).sqrMagnitude;
					//widthOfRightSideOfFace = new Vector2(points[16].x - points[28].x, 1).sqrMagnitude;
					//widthOfFace = widthOfLeftSideOfFace + widthOfRightSideOfFace;

					//distanceOfLeftCheek = new Vector2(points[3].x - points[2].x, 1).sqrMagnitude;
					//distanceOfRightCheek = new Vector2(points[14].x - points[13].x, 1).sqrMagnitude;

					// =====================================================================================
					// Evaluate head rotation
					//Temp = (float)(widthOfRightSideOfFace / widthOfLeftSideOfFace);
					//if (Temp >= 1) {
					//	Temp = (((Mathf.InverseLerp(1f, 10f, Temp))) * -1);
					//} else {
					//	Temp = (1 - (Mathf.InverseLerp(0.45f, 1f, Temp)));
					//}
					//expressionPlayer.headLeft_Right = (float)Math.Round((Temp - 0.15f),2);
					
					
					// =====================================================================================
					// Evaluate head tilt
					//Temp = (float)(heightOfRightSideOfFace / heightOfLeftSideOfFace);					
					//if (Temp >= 1) {
					//	Temp = (((Mathf.InverseLerp(1f, 1.5f, Temp))) * -1);
					//} else {
					//	Temp = (1 - (Mathf.InverseLerp(0.5f, 1f, Temp)));
					//}
					//expressionPlayer.headTiltLeft_Right = (float)(Temp);


					// =====================================================================================
					// Evaluate head up/down
					//Temp = (float) (widthOfEyeSockets / LengthOfNose); //((points[2].y - points[30].y) / 100);
					//Temp = ((Mathf.InverseLerp(1.6f, 3f, Temp)));
					//if (Math.Abs(Temp - last_HeadUPDown) > 0.05)
					//{
					//	last_HeadUPDown = expressionPlayer.headUp_Down;
					//expressionPlayer.headUp_Down = (float)(Math.Round(Temp,2) - 0.2);
					//}

					// =====================================================================================
					// Calculate Action Units
					// =====================================================================================
					
					
					// =====================================================================================
					// Brows (AU1 / AU2 / AU4 )
					/*
					AU1_r = (float)Math.Atan2(points[17].y - points[20].y, points[20].x - points[17].x);
					AU1_l = (float)Math.Atan2(points[26].y - points[23].y, points[26].x - points[23].x);
					AU1 = AU1_r + AU1_l;
					*/
					
					AU1 = (float)(Math.Atan2(points[70].y - points[66].y, points[66].x - points[70].x) + Math.Atan2(points[300].y - points[296].y, points[300].x - points[296].x));
					AU1 += (float)(Math.Atan2(points[46].y - points[65].y, points[65].x - points[46].x) + Math.Atan2(points[276].y - points[295].y, points[276].x - points[295].x));
					AU1 += (float)(Math.Atan2(points[225].y - points[222].y, points[222].x - points[225].x) + Math.Atan2(points[445].y - points[442].y, points[445].x - points[442].x));
					
					if (refine_AU_extents) {
						if (AU1 < AU1_low && AU1 != 0f) { AU1_low = AU1;}
						if (AU1 > AU1_high) { AU1_high = AU1;}
					}
					
					AU1 = (float)Math.Round(Mathf.InverseLerp(AU1_low,AU1_high,AU1),3);
					expressionPlayer.browsIn = (float)Math.Round(Mathf.Lerp(expressionPlayer.browsIn,(1f - AU1),AU1_lowPassFilter),2);			
					
					
					//Debug.Log("counter:" + counter + " AU1=" + (AU1));

					//AU2_r = (float)Math.Atan2(points[17].y - points[18].y, points[18].x - points[17].x);
					//AU2_l = (float)Math.Atan2(points[26].y - points[25].y,points[26].x - points[25].x);
					
					AU2_r = (float)(Math.Atan2(points[70].y - points[63].y, points[63].x - points[70].x) + Math.Atan2(points[46].y - points[53].y, points[53].x - points[46].x) + Math.Atan2(points[225].y - points[224].y, points[224].x - points[225].x));
					AU2_l = (float)(Math.Atan2(points[300].y - points[293].y, points[300].x - points[293].x) + Math.Atan2(points[276].y - points[283].y, points[276].x - points[283].x) + Math.Atan2(points[445].y - points[444].y, points[445].x - points[444].x));		
					
					if (refine_AU_extents) {
						if (AU2_r < AU2_r_low && AU2_r != 0f) { AU2_r_low = AU2_r;}
						if (AU2_r > AU2_r_high) { AU2_r_high = AU2_r;}					
					}
					AU2_r = Mathf.InverseLerp(AU2_r_low,AU2_r_high,AU2_r);

					if (refine_AU_extents) {
						if (AU2_l < AU2_l_low && AU2_l != 0f) { AU2_l_low = AU2_l;}
						if (AU2_l > AU2_l_high) { AU2_l_high = AU2_l;}
					}
					AU2_l = Mathf.InverseLerp(AU2_l_low,AU2_l_high,AU2_l);
					
					Temp = (float)Math.Round(-1 + (AU2_l * 2),4);
					expressionPlayer.leftBrowUp_Down = (float)Math.Round(Mathf.Lerp(expressionPlayer.leftBrowUp_Down,Temp,AU2_lowPassFilter),2);
					Temp = (float)Math.Round(-1 + (AU2_r * 2),4);
					expressionPlayer.rightBrowUp_Down = (float)Math.Round(Mathf.Lerp(expressionPlayer.rightBrowUp_Down,Temp,AU2_lowPassFilter),2);
					AU2 = (float)(Math.Round((AU2_l + AU2_r),3));

			
					//AU4_r = (float)Math.Atan2(points[39].y - points[21].y, points[21].x - points[39].x);
					//AU4_l = (float)Math.Atan2(points[42].y - points[22].y,points[42].x - points[22].x);					
					AU4_r = (float)(Math.Atan2(points[133].y - points[55].y, points[55].x - points[133].x) + Math.Atan2(points[153].y - points[55].y, points[55].x - points[153].x) + Math.Atan2(points[158].y - points[55].y, points[55].x - points[158].x));
					AU4_l = (float)(Math.Atan2(points[362].y - points[285].y,points[362].x - points[285].x) + Math.Atan2(points[380].y - points[285].y,points[380].x - points[285].x) + Math.Atan2(points[385].y - points[285].y,points[385].x - points[285].x));	
									
					AU4 = AU4_l + AU4_r; // Mathf.Abs(AU4_l + AU4_r);
					if (refine_AU_extents) {
						if (AU4 < AU4_low && AU4 != 0f) { AU4_low = AU4;}
						if (AU4 > AU4_high) { AU4_high = AU4;}
					}
					AU4 = (float)Math.Round(Mathf.InverseLerp(AU4_low,AU4_high,AU4),3);
					Temp = (-1f + (AU4 * 2));
					expressionPlayer.midBrowUp_Down = (float)Math.Round(Mathf.Lerp(expressionPlayer.midBrowUp_Down,Temp,AU4_lowPassFilter),2);			
					
					
					// =====================================================================================
					// AU6 - Cheek Raiser - Connected to Left Cheek Squint (-1 to 0) and Smile
					/*
					AU6_r = (float)(distance(points[48].x,points[48].y,points[66].x,points[66].y) / distance(points[48].x,points[48].y,points[51].x,points[51].y)); 
					AU6_l = (float)(distance(points[66].x,points[66].y,points[54].x,points[54].y) / distance(points[51].x,points[51].y,points[54].x,points[54].y));

					if (refine_AU_extents) {
						if (AU6_r < AU6_r_low && AU6_r != 0f) { AU6_r_low = AU6_r;}
						if (AU6_r > AU6_r_high) { AU6_r_high = AU6_r;}
					}
					AU6_r = (float)Math.Round(Mathf.InverseLerp(AU6_r_low,AU6_r_high,AU6_r),1);

					if (refine_AU_extents) {
						if (AU6_l < AU6_l_low && AU6_l != 0f) { AU6_l_low = AU6_l;}
						if (AU6_l > AU6_l_high) { AU6_l_high = AU6_l;}
					}
					AU6_l = (float)Math.Round(Mathf.InverseLerp(AU6_l_low,AU6_l_high,AU6_l),1);

					expressionPlayer.rightMouthSmile_Frown = (float)Math.Round(Mathf.Lerp(expressionPlayer.rightMouthSmile_Frown,AU6_r,AU6_lowPassFilter),2);
					expressionPlayer.leftMouthSmile_Frown = (float)Math.Round(Mathf.Lerp(expressionPlayer.leftMouthSmile_Frown,AU6_l,AU6_lowPassFilter),2);						
					AU6 = (AU6_r + AU6_l)/2;				
					*/

					AU6 = (float)((distance(points[216].x,points[216].y,points[12].x,points[12].y) + distance(points[12].x,points[12].y,points[436].x,points[436].y)) / (distance(points[74].x,points[74].y,points[13].x,points[13].y) + distance(points[13].x,points[13].y,points[304].x,points[304].y))); 

					if (refine_AU_extents) {
						if (AU6 < AU6_low && AU6 != 0f) { AU6_low = AU6;}
						if (AU6 > AU6_high) { AU6_high = AU6;}
					}
					AU6 = (float)Math.Round(Mathf.InverseLerp(AU6_low,AU6_high,AU6),3);

					expressionPlayer.rightMouthSmile_Frown = (float)(Math.Round(Mathf.Lerp(expressionPlayer.rightMouthSmile_Frown,AU6,AU6_lowPassFilter),2));
					expressionPlayer.leftMouthSmile_Frown = (float)(Math.Round(Mathf.Lerp(expressionPlayer.leftMouthSmile_Frown,AU6,AU6_lowPassFilter),2));						
					
					
					// =====================================================================================
					// AU15 - Lip Corner Depressor (use leftLowerLipUp_Down & rightLowerLipUp_Down  ---- or Frown?)
					//AU15_r = (float)(Math.Atan2(points[48].y - points[60].y, points[60].x - points[48].x));
					//AU15_l = (float)(Math.Atan2(points[54].y - points[64].y,points[54].x - points[64].x));
					AU15_r = (float)(Math.Atan2(points[43].y - points[76].y, points[76].x - points[43].x));
					AU15_l = (float)(Math.Atan2(points[273].y - points[306].y,points[273].x - points[306].x));
					//Debug.Log("counter:" + counter + " AU15_r=" + AU15_r + " AU15_l=" + AU15_l);						

					AU15_r = (float)Math.Round(Mathf.InverseLerp(AU15_r_low,AU15_r_high,AU15_r),4);
					if (AU6 < 0.1 && AU15_r > 0) {
						Temp = (0f - AU15_r);
						expressionPlayer.leftMouthSmile_Frown = (float)Math.Round(Mathf.Lerp(expressionPlayer.leftMouthSmile_Frown,Temp,AU15_lowPassFilter),2);	
						//Debug.Log("counter:" + counter + " AU15_r=" + AU15_r);
					}

					AU15_l = (float)Math.Round((Mathf.InverseLerp(AU15_l_low,AU15_l_high,AU15_l)),4);			
					if (AU6 < 0.1 && AU15_l > 0) {
						Temp = (0f - AU15_l);
						expressionPlayer.rightMouthSmile_Frown = (float)Math.Round(Mathf.Lerp(expressionPlayer.rightMouthSmile_Frown,Temp,AU15_lowPassFilter),2);	
						//Debug.Log("counter:" + counter + " AU15_l=" + AU15_l);
					}	
					AU15 = (float)(Math.Round((AU15_r + AU15_l),3));	
					//Debug.Log("counter:" + counter + " AU15=" + AU15);
					//Debug.Log("counter:" + counter + " AU15=" + AU15 + " AU15_r=" + AU15_r + " AU15_l=" + AU15_l);						


					// =====================================================================================
					// AU25/AU26 - Jaw Drop - connected to Mouth Open
					//Temp = (float)((distance(points[61].x,points[61].y,points[67].x,points[67].y) + distance(points[62].x,points[62].y,points[66].x,points[66].y) + distance(points[63].x,points[63].y,points[65].x,points[65].y)));
					//Temp2 = (float)(distance(points[36].x,points[36].y,points[45].x,points[45].y));
					Temp = (float)((distance(points[81].x,points[81].y,points[178].x,points[178].y) + distance(points[13].x,points[13].y,points[14].x,points[14].y) + distance(points[311].x,points[311].y,points[402].x,points[402].y)));
					Temp2 = (float)(distance(points[33].x,points[33].y,points[263].x,points[263].y));
					AU26 = (float)(Temp / Temp2);
					
					if (refine_AU_extents) {
						if (AU26 < AU26_low && AU26 != 0f) { AU26_low = AU26;}
						if (AU26 > AU26_high) { AU26_high = AU26;}
					}
					AU26 = (float)Math.Round(Mathf.InverseLerp(AU26_low,AU26_high,AU26),2);					
					
					expressionPlayer.jawOpen_Close = (float)Math.Round(Mathf.Lerp(expressionPlayer.jawOpen_Close,AU26,AU26_lowPassFilter),1);	
					if (AU26 > 0.1) { AU25 = 1; }


					// =====================================================================================
					// AU23 - Lip Tightener (mouthNarrow_Pucker)
					//AU23 = (float)(Math.Atan2(points[49].y - points[50].y, points[50].x - points[49].x) + Math.Atan2(points[53].y - points[52].y,points[53].x - points[52].x) + Math.Atan2(points[61].y - points[49].y,points[61].x - points[49].x) + Math.Atan2(points[63].y - points[53].y,points[53].x - points[63].x) + Math.Atan2(points[58].y - points[59].y,points[58].x - points[59].x) + Math.Atan2(points[56].y - points[55].y,points[55].x - points[56].x) + Math.Atan2(points[60].y - points[51].y,points[51].x - points[60].x) + Math.Atan2(points[64].y - points[51].y,points[64].x - points[51].x) + Math.Atan2(points[57].y - points[60].y,points[57].x - points[60].x) + Math.Atan2(points[57].y - points[64].y,points[64].x - points[57].x) + Math.Atan2(points[62].y - points[49].y,points[62].x - points[49].x) + Math.Atan2(points[62].y - points[53].y,points[53].x - points[62].x) + Math.Atan2(points[57].y - points[60].y,points[57].x - points[60].x) + Math.Atan2(points[57].y - points[64].y,points[64].x - points[57].x));
					AU23 = (float)(pos_atan2(points[73].y - points[37].y, points[37].x - points[73].x) + pos_atan2(points[303].y - points[267].y,points[303].x - points[267].x) + pos_atan2(points[38].y - points[73].y,points[38].x - points[73].x) + pos_atan2(points[268].y - points[303].y,points[303].x - points[268].x) + pos_atan2(points[84].y - points[180].y,points[84].x - points[180].x) + pos_atan2(points[314].y - points[404].y,points[404].x - points[314].x) + pos_atan2(points[76].y - points[0].y,points[0].x - points[76].x) + pos_atan2(points[306].y - points[0].y,points[306].x - points[0].x) + pos_atan2(points[17].y - points[76].y,points[17].x - points[76].x) + pos_atan2(points[17].y - points[306].y,points[306].x - points[17].x) + pos_atan2(points[12].y - points[73].y,points[12].x - points[73].x) + pos_atan2(points[12].y - points[303].y,points[303].x - points[12].x) + pos_atan2(points[17].y - points[76].y,points[17].x - points[76].x) + pos_atan2(points[17].y - points[306].y,points[306].x - points[17].x));		
					
					if (refine_AU_extents) {
						if (AU23 < AU23_low && AU23 != 0f) { AU23_low = AU23;}
						if (AU23 > AU23_high && AU23 != 0f) { AU23_high = AU23;}					
					}
					AU23 = (float)(Math.Round(Mathf.InverseLerp(AU23_low,AU23_high,AU23),3));	
					//Debug.Log("counter:" + counter + " AU23=" + AU23);					

					expressionPlayer.mouthNarrow_Pucker = (float)(1f - AU23);	
					//Temp = (float)(-1 + Math.Round((AU23 * 2),2));
					//expressionPlayer.mouthNarrow_Pucker = (float)Math.Round(Mathf.Lerp(expressionPlayer.mouthNarrow_Pucker,Temp,AU23_lowPassFilter),1);								

					// =====================================================================================
					// AU7 - Eyes
					//AU7_r = (distance(points[37].x,points[37].y,points[41].x,points[41].y) + distance(points[38].x,points[38].y,points[40].x,points[40].y)) / (2 * distance(points[36].x,points[36].y,points[39].x,points[39].y));
					//AU7_l = (distance(points[43].x,points[43].y,points[47].x,points[47].y) + distance(points[44].x,points[44].y,points[46].x,points[46].y)) / (2 * distance(points[42].x,points[42].y,points[45].x,points[45].y));
					AU7_r = (distance(points[159].x,points[159].y,points[145].x,points[145].y) + distance(points[159].x,points[159].y,points[145].x,points[145].y)) / (2 * distance(points[130].x,points[130].y,points[243].x,points[243].y));
					AU7_l = (distance(points[386].x,points[386].y,points[374].x,points[374].y) + distance(points[386].x,points[386].y,points[374].x,points[374].y)) / (2 * distance(points[463].x,points[463].y,points[359].x,points[359].y));
				
					if (refine_AU_extents) {
						if (AU7_r < AU7_r_low && AU7_r != 0f) { AU7_r_low = AU7_r;}
						if (AU7_r > AU7_r_high) { AU7_r_high = AU7_r;}
					}
					AU7_r = (float)Mathf.InverseLerp(AU7_r_low,(AU7_r_high - 0.01f),AU7_r);					

					if (refine_AU_extents) {
						if (AU7_l < AU7_l_low && AU7_l != 0f) { AU7_l_low = AU7_l;}
						if (AU7_l > AU7_l_high) { AU7_l_high = AU7_l;}
					}
					AU7_l = (float)Mathf.InverseLerp(AU7_l_low,(AU7_l_high - 0.01f),AU7_l);					
					AU7 = (float)Math.Round(((AU7_r + AU7_l)/2),3); 

					Temp = (float)(-1f + (AU7_r * 2));			
					expressionPlayer.rightEyeOpen_Close = (float)Math.Round(Mathf.Lerp(expressionPlayer.rightEyeOpen_Close,Temp,AU7_lowPassFilter),1);
					Temp = (float)(-1f + (AU7_l * 2));
					expressionPlayer.leftEyeOpen_Close = (float)Math.Round(Mathf.Lerp(expressionPlayer.rightEyeOpen_Close,Temp,AU7_lowPassFilter),1);					
					
					
					// =====================================================================================
					// AU9 - Nose Wrinkler
					//AU9 = (float)(Math.Atan2(points[50].y - points[33].y, points[33].x - points[50].x) + Math.Atan2(points[52].y - points[33].y,points[52].x - points[33].x));
					AU9 = (float)(Math.Atan2(points[37].y - points[2].y, points[2].x - points[37].x) + Math.Atan2(points[267].y - points[2].y,points[267].x - points[2].x));
					
					//AU9 = Mathf.Abs(AU9);

					if (refine_AU_extents) {
						if (AU9 < AU9_low && AU9 != 0f) { AU9_low = AU9;}
						if (AU9 > AU9_high) { AU9_high = AU9;}
					}
					AU9 = (float)Math.Round(Mathf.InverseLerp(AU9_low,AU9_high,AU9),3);
					//expressionPlayer.noseSneer = (float)(1f - AU9);
					Temp = (float)(1f - AU9);
					expressionPlayer.noseSneer = (float)Math.Round(Mathf.Lerp(expressionPlayer.noseSneer,Temp,AU9_lowPassFilter),2);


					// =====================================================================================
					// AU20 - Lip Stretcher
					//AU20 = (float)(Math.Atan2(points[59].y - points[65].y, points[65].x - points[59].x) + Math.Atan2(points[55].y - points[67].y, points[55].x - points[67].x) + Math.Atan2(points[59].y - points[66].y, points[66].x - points[59].x) + Math.Atan2(points[59].y - points[67].y, points[67].x - points[59].x) + Math.Atan2(points[55].y - points[65].y, points[55].x - points[65].x));
					
					AU20 = (float)(Math.Atan2(points[180].y - points[317].y, points[317].x - points[180].x) + Math.Atan2(points[404].y - points[87].y, points[404].x - points[87].x)); 
					AU20 += (float)(Math.Atan2(points[180].y - points[87].y, points[87].x - points[180].x) + Math.Atan2(points[404].y - points[317].y, points[404].x - points[317].x));
					AU20 += (float)(Math.Atan2(points[73].y - points[267].y, points[267].x - points[73].x) + Math.Atan2(points[303].y - points[37].y, points[303].x - points[37].x));
					AU20 += (float)(Math.Atan2(points[73].y - points[37].y, points[37].x - points[73].x) + Math.Atan2(points[303].y - points[267].y, points[303].x - points[267].x));					
					AU20 = (float)(Math.Round(AU20,2));
					//Debug.Log("counter:" + counter + " AU20=" + AU20);
					if (refine_AU_extents) {
						if (AU20 < AU20_low && AU20 != 0f) { AU20_low = AU20;}
						if (AU20 > AU20_high) { AU20_high = AU20;}
					}
					AU20 = (float)(Math.Round(Mathf.InverseLerp(AU20_low,AU20_high,AU20),3));
					//Debug.Log("counter:" + counter + " AU20=" + AU20);	
					
					//expressionPlayer.tongueUp_Down = (float)Math.Round(Mathf.Lerp(expressionPlayer.tongueUp_Down,AU20,AU20_lowPassFilter),2);					
					expressionPlayer.tongueUp_Down = (float)Math.Round(AU20,2);					
					
					// =====================================================================================

					//Debug.Log("counter:" + counter + " AU1=" + (AU1) + " low/high=" + AU1_low + " / " + AU1_high);
					//Debug.Log("counter:" + counter + " AU2_r=" + AU2_r + " AU2_l=" + AU2_l);
					//Debug.Log("counter:" + counter + " AU4=" + AU4 + " low=" + AU4_low + "  high=" + AU4_high);
					//Debug.Log("counter:" + counter + " AU6=" + AU6_l + "   " + AU6_r);

					//Debug.Log("counter:" + counter + " AU26=" + AU26);
					//Debug.Log("counter:" + counter + " AU20=" + AU20 + "   - " + AU20_low + "/" + AU20_high + "        ==" + expressionPlayer.mouthNarrow_Pucker);										
					//Debug.Log("counter:" + counter + " AU23=" + AU23 + "   - " + AU23_low + "/" + AU23_high + "        ==" + expressionPlayer.mouthNarrow_Pucker);										
					
					//Debug.Log("counter:" + counter + " AU7_r=" + AU7_r + " low=" + AU7_r_low + "  high=" + AU7_r_high);		

					//Debug.Log("counter:" + counter + " AU9=" + AU9 + " low=" + AU9_low + "  high=" + AU9_high);
					
					faceAU.AU1 = AU1;
					faceAU.AU2 = AU2;
					faceAU.AU4 = AU4;
					faceAU.AU6 = AU6;
					faceAU.AU7 = AU7;
					faceAU.AU9 = AU9;					
					faceAU.AU15 = AU15;
					faceAU.AU20 = AU20;
					faceAU.AU23 = AU23;
					faceAU.AU25 = AU25;
					faceAU.AU26 = AU26;	
		
				}
			}
			return rtn;
        }

		protected virtual float ease(float From, float To, float coeff)	
		{
			//float diff = Math.Abs(From - To);
			//float New = From + (diff * coeff);
			return To;
		}		
		
		protected virtual float distance(float From_X, float From_Y, float To_X, float To_Y)	
		{
			float dist = (float)Math.Sqrt( Math.Pow(To_X - From_X,2) + Math.Pow(To_Y - From_Y,2) );
			return dist;
		}

		protected virtual float pos_atan2(float y, float x)	
		{
			float ret = (float)Math.Atan2(y, x); 
			if (ret < 0) { ret = 0; };
			return ret;
		}
		
		protected virtual float lerp(float x, float a, float b)	
		{
			float ret = (x - a) / (b - a);
			if (ret < 0) { ret = 0; };
			return (float)Math.Round(ret,2);		
		}		
		
		public void OnExitButtonClick()
		{
			Debug.Log("Quit");
			Application.Quit();
		}

		public void OnRecordButtonClick()
		{
			
			// Toggle button for recording on/off
			if (Start_Recording) {
				
				// Turn off recording
				Start_Recording = false;
				Recording_FirstTime = true;
				RecordButton_Text.text = "Record";
				//recognitionTime = recognitionIntervalTime;	
				recognitionFrames = recognitionIntervalFrames;	
				
				audioSource.Stop();
				tempRecording.Clear();
				Microphone.End(null);
				
				ResetBufferTotals();
				
			} else {
				
				capture.set(Videoio.CAP_PROP_POS_FRAMES, 0);
				currentFrame = 1;
				previousFrame = 0;
				
				// Turn on recording
				Start_Recording = true;
				RecordButton_Text.text = "Stop";
				overallFacialInput.Clear();				
				facialInput.Clear();
				//recognitionTime = recognitionIntervalTime;	
				recognitionFrames = recognitionIntervalFrames;	
				
			}

		}		

		
	}

}