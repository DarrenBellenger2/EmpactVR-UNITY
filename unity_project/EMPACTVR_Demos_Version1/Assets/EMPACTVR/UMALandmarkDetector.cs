#if !(PLATFORM_LUMIN && !UNITY_EDITOR)

using UMA;
using UMA.PoseTools;
//using DlibFaceLandmarkDetectorUMA;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using Unity.Mathematics;

using OpenCVForUnity.UnityUtils.Helper;
using OpenCVForUnity.UnityUtils;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.FaceModule;
//using OpenCVForUnity.ObjdetectModule;
using Rect = OpenCVForUnity.CoreModule.Rect;
using OpenCVForUnity.UtilsModule;
using OpenCVForUnityExample;
using OpenCVForUnity.VideoioModule;
using Mediapipe.Unity.Sample.FaceLandmarkDetection;

namespace LandmarkDetectorUMA
{

    [RequireComponent(typeof(WebCamTextureToMatHelper))]
    public class UMALandmarkDetector : MonoBehaviour
    {

		// Video
        public Slider seekBarSlider;
        Slider.SliderEvent defaultSliderEvent = new Slider.SliderEvent();
        VideoCapture capture;
        bool isPlaying = false;
        bool shouldUpdateVideoFrame = false;
        long prevFrameTickCount;
        long currentFrameTickCount;
		Mat rgbVideoMat;
        Texture2D Video_texture;		
		protected static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/01-01-05-01-01-01-01_1.mp4";

        public Dropdown requestedModeDropdown;
		public int requestedMode = 0;
		bool video_capture_initialised = false;
		const int K_Webcam = 0;
		const int K_Video = 1;

        public bool isDebugMode = false;			
        public UMAAvatarBase targetUMA = null;

        [SerializeField, TooltipAttribute("Set the name of the device to use.")]
        public string requestedDeviceName = null;

        [SerializeField, TooltipAttribute("Set the width of WebCamTexture.")]
        public int requestedWidth = 320;

        [SerializeField, TooltipAttribute("Set the height of WebCamTexture.")]
        public int requestedHeight = 240;

        [SerializeField, TooltipAttribute("Set FPS of WebCamTexture.")]
        public int requestedFPS = 60;
		
        [SerializeField, TooltipAttribute("Set fames to skip before landmarks updated.")]
        public int frameSkipping = 0;	
		//List<Vector2> landmark_points;
		List<UnityEngine.Rect> detectResult;

		[Space]
		//[SerializeField] ResourceSet _resources = null;	
		[SerializeField] RawImage _faceUI = null;
		[SerializeField] Texture _screenshot = null;	
		//FacePipeline _pipeline;
		
        Mat grayMat;
        Texture2D WebCam_texture;
        MatOfRect faces;
        WebCamTextureToMatHelper webCamTextureToMatHelper;
        FpsMonitor fpsMonitor;

        ExpressionPlayer expressionPlayer;
        int counter = 0;
        int webcam_width, webcam_height;
        float last_HeadUPDown = 0f;

		const int numFinalLandmarkPoints = 68;	


		
        float AU1_low = 1.2f;
        float AU1_high = 0.1f;

		float AU2_l_low	= 5f; // 1.15f;
		float AU2_l_high =0.0001f; // 0.3f; 
		
		float AU2_r_low	= 5f; // 1.15f;
		float AU2_r_high = 0.0001f; // 0.3f;		
	
		
        float AU4_low = 5f; //3f
        float AU4_high = 0.0001f; //1.2f;

        //float AU6_low = 5f; //1.16f
        //float AU6_high = 0.0001f; //0.1f		
					
		float AU6_l_low	= 5f; // 0.35f;
		float AU6_l_high = 0.0001f; // 0.3f; 
		
		float AU6_r_low	= 5f; // 0.35f;
		float AU6_r_high = 0.0001f; // 0.3f;


		float AU7_l_low	= 0.7f;  // 0.35f;
		float AU7_l_high = 0.1f; // 0.3f; 
		
		float AU7_r_low	= 0.7f;  // 0.35f;
		float AU7_r_high = 0.1f; // 0.3f;
			
        float AU9_low = 2.5f;  // 1f
        float AU9_high = 1f; // 2.5f;

		float AU15_l_low = -0.25f;  // -0.5f
        float AU15_l_high = 0.5f; // 1f;

		float AU15_r_low = -0.25f;  // -0.5f
        float AU15_r_high = 0.5f; // 1f;
		
		float AU23_low = 10f; //10f
		float AU23_high = -10f; // -10f;	
			
		//float AU26_low = 2f;  // 0.1f
        //float AU26_high = 0.01f; // 1f;

		//public NNModel modelAsset;
		//private Model m_RuntimeModel;

		
        //protected float distanceOfLeftEyeHeight;
        //protected float distanceOfRightEyeHeight;
        //protected float distanceOfNoseHeight;
        //protected float distanceBetweenLeftPupliAndEyebrow;
        //protected float distanceBetweenRightPupliAndEyebrow;
        //protected float distanceOfMouthHeight;
        //protected float distanceOfMouthWidth;
        //protected float distanceBetweenEyes;	

        // ============================================================================================
        //FaceLandmarkDetector faceLandmarkDetector;
        //string dlibShapePredictorFileName = "sp_human_face_68.dat";
        //string dlibShapePredictorFilePath;
        //double[] landmark_array;

		//private CalculatorGraph _graph;
		//private OutputStream<ImageFrame> _outputVideoStream;
		//private OutputStream<List<NormalizedLandmarkList>> _multiFaceLandmarksStream;
		//private ResourceManager _resourceManager;

		

#if UNITY_WEBGL && !UNITY_EDITOR
        IEnumerator getFilePath_Coroutine;
#endif

        // Use this for initialization
        void Start()
        {
            fpsMonitor = GetComponent<FpsMonitor>();
            webCamTextureToMatHelper = gameObject.GetComponent<WebCamTextureToMatHelper>();
			//dlibShapePredictorFileName = "sp_human_face_68.dat";
			//landmark_points = new List<Vector2>();
			//detectResult = new List<UnityEngine.Rect>();		
			
#if UNITY_WEBGL && !UNITY_EDITOR
            getFilePath_Coroutine = Utils.getFilePathAsync(dlibShapePredictorFileName, (result) =>
            {
                getFilePath_Coroutine = null;

                dlibShapePredictorFilePath = result;
                Run();
            });
            StartCoroutine(getFilePath_Coroutine);
#else
			
			//_pipeline = new FacePipeline(_resources);	
			//dlibShapePredictorFilePath = DlibFaceLandmarkDetector.UnityUtils.Utils.getFilePath(dlibShapePredictorFileName);
            Run();
#endif
        }

        private void Run()
        {
			
            //if true, The error log of the Native side OpenCV will be displayed on the Unity Editor Console.
            OpenCVForUnity.UnityUtils.Utils.setDebugMode(true);

			//if (string.IsNullOrEmpty(dlibShapePredictorFilePath)) {
            //    Debug.LogError("shape predictor file does not exist. Please copy from “DlibFaceLandmarkDetector/StreamingAssets/” to //“Assets/StreamingAssets/” folder. ");
            //}
            //faceLandmarkDetector = new FaceLandmarkDetector(dlibShapePredictorFilePath);
					
            //Initialize();
#if UNITY_ANDROID && !UNITY_EDITOR
            // Avoids the front camera low light issue that occurs in only some Android devices (e.g. Google Pixel, Pixel2).
            webCamTextureToMatHelper.avoidAndroidFrontCameraLowLightIssue = true;
#endif
            webCamTextureToMatHelper.Initialize();
        }	

        public void OnRequestedModeDropdownValueChanged(int result)
        {
			
            if (requestedMode == K_Webcam)
            {
                requestedMode = K_Video;
				video_capture_initialised = false;
				shouldUpdateVideoFrame = false;
				
            } else {
                requestedMode = K_Webcam;
				gameObject.GetComponent<Renderer>().material.mainTexture = WebCam_texture;
			}
			
        }

        public void OnPlayButtonClick()
        {
            isPlaying = true;
        }

        public void OnPauseButtonClick()
        {
            isPlaying = false;
        }

        public void OnJumpAheadButtonClick()
        {
            int courentFrame = (int)capture.get(Videoio.CAP_PROP_POS_FRAMES) + 50;
            if (courentFrame >= capture.get(Videoio.CAP_PROP_FRAME_COUNT))
            {
                courentFrame = (int)capture.get(Videoio.CAP_PROP_FRAME_COUNT);
            }

            capture.set(Videoio.CAP_PROP_POS_FRAMES, courentFrame);
        }

        public void OnJumpBackButtonClick()
        {
            int courentFrame = (int)capture.get(Videoio.CAP_PROP_POS_FRAMES) - 50;
            if (courentFrame <= 0)
            {
                courentFrame = 0;
            }

            capture.set(Videoio.CAP_PROP_POS_FRAMES, courentFrame);
        }

        public void OnSeekBarSliderValueChanged()
        {
            bool supported = capture.set(Videoio.CAP_PROP_POS_AVI_RATIO, seekBarSlider.value);

            if (!supported)
            {
                capture.set(Videoio.CAP_PROP_POS_FRAMES, (int)(seekBarSlider.value * capture.get(Videoio.CAP_PROP_FRAME_COUNT)));
            }
        }

        private IEnumerator WaitFrameTime()
        {
            double videoFPS = (capture.get(Videoio.CAP_PROP_FPS) <= 0) ? 10.0 : capture.get(Videoio.CAP_PROP_FPS);
            float frameTime_sec = (float)(1000.0 / videoFPS / 1000.0);
            WaitForSeconds wait = new WaitForSeconds(frameTime_sec);
            prevFrameTickCount = currentFrameTickCount = Core.getTickCount();

            capture.grab();

            while (true)
            {
                if (isPlaying)
                {
                    shouldUpdateVideoFrame = true;

                    prevFrameTickCount = currentFrameTickCount;
                    currentFrameTickCount = Core.getTickCount();

                    yield return wait;
                }
                else
                {
                    yield return null;
                }
            }
        }
		
        /// <summary>
        /// Raises the web cam texture to mat helper initialized event.
        /// </summary>
        public void OnWebCamTextureToMatHelperInitialized()
        {
            //Debug.Log("OnWebCamTextureToMatHelperInitialized");

            Mat webCamTextureMat = webCamTextureToMatHelper.GetMat();

            WebCam_texture = new Texture2D(webCamTextureMat.cols(), webCamTextureMat.rows(), TextureFormat.RGBA32, false);
            OpenCVForUnity.UnityUtils.Utils.fastMatToTexture2D(webCamTextureMat, WebCam_texture);

            gameObject.GetComponent<Renderer>().material.mainTexture = WebCam_texture;
            //Debug.Log("Screen.width " + Screen.width + " Screen.height " + Screen.height + " Screen.orientation " + Screen.orientation);

            if (fpsMonitor != null)
            {
                fpsMonitor.Add("width", webCamTextureToMatHelper.GetWidth().ToString());
                fpsMonitor.Add("height", webCamTextureToMatHelper.GetHeight().ToString());
                fpsMonitor.Add("orientation", Screen.orientation.ToString());
            }

            float webcam_width = 640; //webCamTextureMat.width();
            float webcam_height = 480; //webCamTextureMat.height();

            float widthScale = (float)Screen.width / webcam_width;
            float heightScale = (float)Screen.height / webcam_height;
            if (widthScale < heightScale)
            {
                Camera.main.orthographicSize = (webcam_width * (float)Screen.height / (float)Screen.width) / 2;
            }
            else
            {
                Camera.main.orthographicSize = webcam_height / 2;
            }

            grayMat = new Mat(webCamTextureMat.rows(), webCamTextureMat.cols(), CvType.CV_8UC1);
            faces = new MatOfRect();
			
        }

		/// <summary>
        /// Raises the web cam texture to mat helper disposed event.
        /// </summary>
        public void OnWebCamTextureToMatHelperDisposed()
        {
            Debug.Log("OnWebCamTextureToMatHelperDisposed");
		
            if (grayMat != null)
                grayMat.Dispose();


            if (WebCam_texture != null)
            {
                Texture2D.Destroy(WebCam_texture);
                WebCam_texture = null;
            }
			
            if (Video_texture != null)
            {
                Texture2D.Destroy(Video_texture);
                Video_texture = null;
            }			

            if (faces != null)
                faces.Dispose();			
			
        }
		
        /// <summary>
        /// Raises the web cam texture to mat helper error occurred event.
        /// </summary>
        /// <param name="errorCode">Error code.</param>
        public void OnWebCamTextureToMatHelperErrorOccurred(WebCamTextureToMatHelper.ErrorCode errorCode)
        {
            Debug.Log("OnWebCamTextureToMatHelperErrorOccurred " + errorCode);
        }
		

        // Update is called once per frame
        void Update()
        {
	
			counter++;	
			if (requestedMode == K_Webcam)
			{

				if (webCamTextureToMatHelper.IsPlaying() && webCamTextureToMatHelper.DidUpdateThisFrame())
				{
					//_pipeline.ProcessImage(WebCam_texture);		
					Mat rgbaMat = webCamTextureToMatHelper.GetMat();
					ProcessFace(rgbaMat);
				}

			} else {
				
				ProcessVideo();
				
			}
			
        }

		void ProcessVideo ()	
        {

			// Initialise video
			if (!video_capture_initialised) 
			{
				
				rgbVideoMat = new Mat();
				
				capture = new VideoCapture();
				capture.open(OpenCVForUnity.UnityUtils.Utils.getFilePath(VIDEO_FILENAME));
				video_capture_initialised = true;
				
				capture.grab();
				capture.retrieve(rgbVideoMat);
				Video_texture = new Texture2D(rgbVideoMat.cols(), rgbVideoMat.rows(), TextureFormat.RGB24, false);
			
				gameObject.GetComponent<Renderer>().material.mainTexture = Video_texture;
				
				StartCoroutine("WaitFrameTime");
				capture.set(Videoio.CAP_PROP_POS_FRAMES, 0);
				isPlaying = true;							
			}

            if (isPlaying && shouldUpdateVideoFrame)
            {
                shouldUpdateVideoFrame = false;

                //Loop play
                if (capture.get(Videoio.CAP_PROP_POS_FRAMES) >= capture.get(Videoio.CAP_PROP_FRAME_COUNT))
                    capture.set(Videoio.CAP_PROP_POS_FRAMES, 0);

                if (capture.grab())
                {

					capture.retrieve(rgbVideoMat);
                    Imgproc.cvtColor(rgbVideoMat, rgbVideoMat, Imgproc.COLOR_BGR2RGB);

					OpenCVForUnity.UnityUtils.Utils.fastMatToTexture2D(rgbVideoMat, Video_texture);
					//_pipeline.ProcessImage(Video_texture);				
					
					ProcessFace(rgbVideoMat);
										
                    //var tmp = seekBarSlider.onValueChanged;
                    //seekBarSlider.onValueChanged = defaultSliderEvent;
                    //seekBarSlider.value = (float)capture.get(Videoio.CAP_PROP_POS_AVI_RATIO);
                    //seekBarSlider.onValueChanged = tmp;
                }

			}
			

		}
		
		void ProcessFace (Mat FaceMat)	
        {
			//bool detectLandmarks = false;
			//Mat tempMat;
			//List<Vector2> temp_points;
			//temp_points = new List<Vector2>();
			//List<Vector2> landmark_points;
			
			//landmark_points = new List<Vector2>();

			//Debug.Log("counter:" + counter);
			//Debug.Log("counter:" + counter + "    frameSkipping=" + frameSkipping + "  - " + (counter % frameSkipping));
			/*
			if ((frameSkipping == 0) || (counter % frameSkipping) == 0) { 
			
				detectLandmarks = true;
				OpenCVForUnityUtils.SetImage(faceLandmarkDetector, FaceMat);
				detectResult = faceLandmarkDetector.Detect();

			}


			if ( detectResult.Count > 0) {

				temp_points = faceLandmarkDetector.DetectLandmark(detectResult[0]);
				//landmark_points.Clear();
				for (int j = 0; j < temp_points.Count; j++) { 
					//landmark_points.Add(new Vector2(temp_points[j].x, (webcam_height - temp_points[j].y)));
					landmark_points.Add(new Vector2(temp_points[j].x, (temp_points[j].y)));
				}
			
			}
			*/
			
			/*
			if ( landmark_points.Count == 68)
			{
				
				Texture2D tex2D = new Texture2D(webcam_width, webcam_height, TextureFormat.RGBA32, false);				
				
				switch (requestedMode)
				{
					case K_Webcam:
						OpenCVForUnity.UnityUtils.Utils.fastMatToTexture2D(FaceMat, WebCam_texture);
						tex2D = WebCam_texture;
						break;
					case K_Video:				
						OpenCVForUnity.UnityUtils.Utils.fastMatToTexture2D(FaceMat, Video_texture); //rgbVideoMat
						tex2D = Video_texture;
						break;					
				}

				if (landmark_points != null && !isDebugMode)
				{
					for (int j = 0; j < landmark_points.Count; j++) { Circle(tex2D, (int)(landmark_points[j].x), (int)(landmark_points[j].y), 1, Color.white); }
				}
				tex2D.Apply();					
				_faceUI.texture = tex2D;				

				CalculateAvatarSettings (landmark_points);			
			}
			*/

			Texture2D tex2D = new Texture2D(webcam_width, webcam_height, TextureFormat.RGBA32, false);
			switch (requestedMode)
			{
				case K_Webcam:
					OpenCVForUnity.UnityUtils.Utils.fastMatToTexture2D(FaceMat, WebCam_texture);
					tex2D = WebCam_texture;
					break;
				case K_Video:				
					OpenCVForUnity.UnityUtils.Utils.fastMatToTexture2D(FaceMat, Video_texture); //rgbVideoMat
					tex2D = Video_texture;
					break;					
			}
			
			tex2D.Apply();					
			_faceUI.texture = tex2D;				
			
		}	
	
        private List<Vector2> ConvertMediaPipeLandmarksToVectorList(int w, int h, List<Vector2> pts = null)
        {
			//int[] landmarks_68 = new int[] {162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,71,63,105,66,107,336,
			//						296,334,293,301,168,197,5,4,75,97,2,326,305,33,160,158,133,153,144,362,385,387,263,373,
			//						380,61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87};				
			
			// Jawline 0-15
			// Right-eyebrow 16-21
			// Left-eyebrow 22-26
			// Nose 27-35
			// Eyes 36-47
			// Mouth 48-68
			//int[] landmarks_68 = new int[] {162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,
			//						70,63,105,66,193,
			//						417,296,334,293,300,
			//						168,197,5,4,75,97,2,326,305,
			//						33,160,158,133,153,144,362,385,387,263,373,380,
			//						61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87};
			int[] landmarks_68 = new int[] {162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,
									70,63,105,66,55,
									285,296,334,293,300,
									168,197,5,4,75,97,2,326,305,
									33,160,158,133,153,144,362,385,387,263,373,380,
									61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87};
			float2 temp; 
			
            if (pts == null)
            {
                pts = new List<Vector2>();
            }
            pts.Clear();

			for (int i = 0; i < numFinalLandmarkPoints; i++)
			{
				temp = 0; //_pipeline.GetFaceVertex( landmarks_68[i]).xy;
				
				float x = temp[0] * w;
				float y = temp[1] * h;
				pts.Add(new Vector2(x, y));
			}		
			
			// =========================================================================================
			//Debug.Log("Crop=" + _pipeline.FaceCropMatrix);
			
			//var test_index1 = _pipeline.GetFaceVertex( landmarks_68[0]).xy;
			//var test_index2 = _pipeline.GetFaceVertex( landmarks_68[16]).xy;
			//var test_index3 = _pipeline.GetFaceVertex( landmarks_68[8]).xy;
			//Debug.Log("test_index=" + test_index1 + "     " + test_index2 + "     " + test_index3);
			// =========================================================================================			
			
            return pts;
        }
		
		private void Circle(Texture2D tex, int cx, int cy, int r, Color col)
		{
			int x, y, px, nx, py, ny, d;
				 
			for (x = 0; x <= r; x++)
			{
				d = (int)Mathf.Ceil(Mathf.Sqrt(r * r - x * x));
				for (y = 0; y <= d; y++)
				{
					px = cx + x;
					nx = cx - x;
					py = cy + y;
					ny = cy - y;

					tex.SetPixel(px, py, col);
					tex.SetPixel(nx, py, col);

					tex.SetPixel(px, ny, col);
					tex.SetPixel(nx, ny, col);

				}
			}    
		}			
		
		
		
        protected virtual void CalculateAvatarSettings (List<Vector2> points)
        {

			
		    double widthOfLeftEyeSocket;
            double widthOfRightEyeSocket;
            double widthOfEyeSockets;
			double LengthOfNose;


			double widthOfLeftSideOfFace;
            double widthOfRightSideOfFace;
            double widthOfFace;
            //double widthOfBrows;

            double heightOfLeftSideOfFace;
            double heightOfRightSideOfFace;
            double heightOfUpperFace;
            double heightOfLowerFace;
            double distanceOfLeftCheek;
            double distanceOfRightCheek;

            //double distanceOfLeftEyeHeight;
            //double distanceOfRightEyeHeight;
            //double distanceOfNoseHeight;
            //double distanceBetweenLeftPupliAndEyebrow;
            //double distanceBetweenRightPupliAndEyebrow;
            //double distanceOfMouthHeight;
            //double distanceOfMouthWidth;
            //double distanceBetweenEyes;
            //double RightSmileHeight;
            //double LeftSmileHeight;
            //double RightEyeOpenShut;
            //double LeftEyeOpenShut;

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
			float AU23_lowPassFilter = 0.5f;
			float AU26_lowPassFilter = 0.75f;	
		
			if (targetUMA != null)
			{

				expressionPlayer = targetUMA.GetComponent<ExpressionPlayer>();
				if (expressionPlayer == null)
				{
					Debug.Log("No expression player!");
					return;
				}
				else
				{

					// =====================================================================
					// Work out expressions
					// =====================================================================

					//counter++;
					widthOfLeftEyeSocket = distance(points[27].x,points[27].y,points[36].x,points[36].y);
					widthOfRightEyeSocket = distance(points[27].x,points[27].y,points[45].x,points[45].y);
					widthOfEyeSockets = widthOfLeftEyeSocket + widthOfRightEyeSocket;
					LengthOfNose = distance(points[27].x,points[27].y,points[30].x,points[30].y);

					heightOfLeftSideOfFace = distance(0,points[0].y,0,points[8].y);
					heightOfRightSideOfFace = distance(0,points[16].y,0,points[8].y);				

					widthOfLeftSideOfFace = new Vector2(points[28].x - points[0].x, 1).sqrMagnitude;
					widthOfRightSideOfFace = new Vector2(points[16].x - points[28].x, 1).sqrMagnitude;
					widthOfFace = widthOfLeftSideOfFace + widthOfRightSideOfFace;

					//widthOfBrows = new Vector2(points[22].x - points[21].x, 1).sqrMagnitude;

					distanceOfLeftCheek = new Vector2(points[3].x - points[2].x, 1).sqrMagnitude;
					distanceOfRightCheek = new Vector2(points[14].x - points[13].x, 1).sqrMagnitude;


					//distanceOfLeftEyeHeight = new Vector2((points[47].x + points[46].x) / 2 - (points[43].x + points[44].x) / 2, (points[47].y + points[46].y) / 2 - (points[43].y + points[44].y) / 2).sqrMagnitude;
					//distanceOfRightEyeHeight = new Vector2((points[40].x + points[41].x) / 2 - (points[38].x + points[37].x) / 2, (points[40].y + points[41].y) / 2 - (points[38].y + points[37].y) / 2).sqrMagnitude;
					//distanceOfNoseHeight = new Vector2(points[33].x - (points[39].x + points[42].x) / 2, points[33].y - (points[39].y + points[42].y) / 2).sqrMagnitude;

					//distanceBetweenLeftPupliAndEyebrow = new Vector2(points[24].x - (points[42].x + points[45].x) / 2, points[24].y - (points[42].y + points[45].y) / 2).sqrMagnitude;
					//distanceBetweenRightPupliAndEyebrow = new Vector2(points[19].x - (points[39].x + points[36].x) / 2, points[19].y - (points[39].y + points[36].y) / 2).sqrMagnitude;

					//RightEyeOpenShut = new Vector2((points[38].x - points[37].x), ((points[41].y + points[40].y) / 2) - (points[37].y + points[38].y) / 2).sqrMagnitude;
					//RightEyeOpenShut = new Vector2(1, ((points[41].y + points[40].y) / 2) - (points[37].y + points[38].y) / 2).sqrMagnitude;
					//LeftEyeOpenShut = new Vector2(1, ((points[47].y + points[46].y) / 2) - (points[43].y + points[44].y) / 2).sqrMagnitude;

					//distanceOfMouthHeight = new Vector2(points[51].x - points[57].x, points[51].y - points[57].y).sqrMagnitude;
					//distanceOfMouthWidth = new Vector2(points[48].x - points[54].x, points[48].y - points[54].y).sqrMagnitude;
					//distanceBetweenEyes = new Vector2(points[39].x - points[42].x, points[39].y - points[42].y).sqrMagnitude;


					//RightSmileHeight = new Vector2(((points[39].x + points[42].x) / 2) - points[48].x, points[48].y - ((points[39].y + points[42].y) / 2)).sqrMagnitude;
					//LeftSmileHeight = new Vector2(points[54].x - ((points[39].x + points[42].x) / 2), points[54].y - ((points[39].y + points[42].y) / 2)).sqrMagnitude;
		

					// =====================================================================================
					// Evaluate head rotation
					Temp = (float)(widthOfRightSideOfFace / widthOfLeftSideOfFace);
					if (Temp >= 1) {
						Temp = (((Mathf.InverseLerp(1f, 10f, Temp))) * -1);
					} else {
						Temp = (1 - (Mathf.InverseLerp(0.45f, 1f, Temp)));
					}
					//expressionPlayer.headLeft_Right = (float)Math.Round((Temp - 0.15f),2);
					
					
					// =====================================================================================
					// Evaluate head tilt
					Temp = (float)(heightOfRightSideOfFace / heightOfLeftSideOfFace);					
					if (Temp >= 1) {
						Temp = (((Mathf.InverseLerp(1f, 1.5f, Temp))) * -1);
					} else {
						Temp = (1 - (Mathf.InverseLerp(0.5f, 1f, Temp)));
					}
					//expressionPlayer.headTiltLeft_Right = (float)(Temp);


					// =====================================================================================
					// Evaluate head up/down
					Temp = (float) (widthOfEyeSockets / LengthOfNose); //((points[2].y - points[30].y) / 100);
					Temp = ((Mathf.InverseLerp(1.6f, 3f, Temp)));
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
					AU1_r = (float)Math.Atan2(points[17].y - points[20].y, points[20].x - points[17].x);
					AU1_l = (float)Math.Atan2(points[26].y - points[23].y, points[26].x - points[23].x);
					AU1 = AU1_r + AU1_l;
					if (AU1 < AU1_low && AU1 != 0f) { AU1_low = AU1;}
					if (AU1 > AU1_high) { AU1_high = AU1;}
					
					AU1 = (float)Math.Round(Mathf.InverseLerp(AU1_low,AU1_high,AU1),2);
					Temp = (1f - AU1);
					expressionPlayer.browsIn = (float)Math.Round(Mathf.Lerp(expressionPlayer.browsIn,Temp,AU1_lowPassFilter),2);


					AU2_r = (float)Math.Atan2(points[17].y - points[18].y, points[18].x - points[17].x);
					AU2_l = (float)Math.Atan2(points[26].y - points[25].y,points[26].x - points[25].x);
					
					if (AU2_r < AU2_r_low && AU2_r != 0f) { AU2_r_low = AU2_r;}
					if (AU2_r > AU2_r_high) { AU2_r_high = AU2_r;}					
					AU2_r = Mathf.InverseLerp(AU2_r_low,AU2_r_high,AU2_r);
					
					if (AU2_l < AU2_l_low && AU2_l != 0f) { AU2_l_low = AU2_l;}
					if (AU2_l > AU2_l_high) { AU2_l_high = AU2_l;}					
					AU2_l = Mathf.InverseLerp(AU2_l_low,AU2_l_high,AU2_l);
					
					Temp = (-1 + (AU2_l * 2));
					expressionPlayer.leftBrowUp_Down = (float)Math.Round(Mathf.Lerp(expressionPlayer.leftBrowUp_Down,Temp,AU2_lowPassFilter),2);
					Temp = (-1 + (AU2_r * 2));
					expressionPlayer.rightBrowUp_Down = (float)Math.Round(Mathf.Lerp(expressionPlayer.rightBrowUp_Down,Temp,AU2_lowPassFilter),2);
					AU2 = (AU2_l + AU2_r)/2;

			
					AU4_r = (float)Math.Atan2(points[39].y - points[21].y, points[21].x - points[39].x);
					AU4_l = (float)Math.Atan2(points[42].y - points[22].y,points[42].x - points[22].x);					
					AU4 = Mathf.Abs(AU4_l + AU4_r);
					if (AU4 < AU4_low && AU4 != 0f) { AU4_low = AU4;}
					if (AU4 > AU4_high) { AU4_high = AU4;}
					AU4 = (float)Math.Round(Mathf.InverseLerp(AU4_low,AU4_high,AU4),2);
					Temp = (-1f + (AU4 * 2));
					expressionPlayer.midBrowUp_Down = (float)Math.Round(Mathf.Lerp(expressionPlayer.midBrowUp_Down,Temp,AU4_lowPassFilter),2);			
					
					
					// =====================================================================================
					// AU6 - Cheek Raiser - Connected to Left Cheek Squint (-1 to 0) and Smile
					AU6_r = (float)(distance(points[48].x,points[48].y,points[66].x,points[66].y) / distance(points[48].x,points[48].y,points[51].x,points[51].y)); 
					AU6_l = (float)(distance(points[66].x,points[66].y,points[54].x,points[54].y) / distance(points[51].x,points[51].y,points[54].x,points[54].y));

					if (AU6_r < AU6_r_low && AU6_r != 0f) { AU6_r_low = AU6_r;}
					if (AU6_r > AU6_r_high) { AU6_r_high = AU6_r;}
					AU6_r = (float)Math.Round(Mathf.InverseLerp(AU6_r_low,AU6_r_high,AU6_r),1);

					if (AU6_l < AU6_l_low && AU6_l != 0f) { AU6_l_low = AU6_l;}
					if (AU6_l > AU6_l_high) { AU6_l_high = AU6_l;}
					AU6_l = (float)Math.Round(Mathf.InverseLerp(AU6_l_low,AU6_l_high,AU6_l),1);

					expressionPlayer.rightMouthSmile_Frown = (float)Math.Round(Mathf.Lerp(expressionPlayer.rightMouthSmile_Frown,AU6_r,AU6_lowPassFilter),2);
					expressionPlayer.leftMouthSmile_Frown = (float)Math.Round(Mathf.Lerp(expressionPlayer.leftMouthSmile_Frown,AU6_l,AU6_lowPassFilter),2);						
					AU6 = AU6_r + AU6_l;				
					
					
					// =====================================================================================
					// AU15 - Lip Corner Depressor (use leftLowerLipUp_Down & rightLowerLipUp_Down  ---- or Frown?)
					AU15_r = (float)(Math.Atan2(points[48].y - points[60].y, points[60].x - points[48].x));
					AU15_r = (float)(Mathf.InverseLerp(AU15_r_low,AU15_r_high,AU15_r));
					if (AU6_r < 0.1 && AU15_r > 0) {
						Temp = (0f - AU15_r);
						expressionPlayer.leftMouthSmile_Frown = (float)Math.Round(Mathf.Lerp(expressionPlayer.leftMouthSmile_Frown,Temp,AU15_lowPassFilter),2);	
						//Debug.Log("counter:" + counter + " AU15_r=" + AU15_r);
					}

					AU15_l = (float)(Math.Atan2(points[54].y - points[64].y,points[54].x - points[64].x));
					AU15_l = (float)(Mathf.InverseLerp(AU15_l_low,AU15_l_high,AU15_l));			
					if (AU6_l < 0.1 && AU15_l > 0) {
						Temp = (0f - AU15_l);
						expressionPlayer.rightMouthSmile_Frown = (float)Math.Round(Mathf.Lerp(expressionPlayer.rightMouthSmile_Frown,Temp,AU15_lowPassFilter),2);	
						//Debug.Log("counter:" + counter + " AU15_l=" + AU15_l);
					}		


					// =====================================================================================
					// AU25/AU26 - Jaw Drop - connected to Mouth Open
					Temp = (float)((distance(points[61].x,points[61].y,points[67].x,points[67].y) + distance(points[62].x,points[62].y,points[66].x,points[66].y) + distance(points[63].x,points[63].y,points[65].x,points[65].y)));
					Temp2 = (float)(distance(points[36].x,points[36].y,points[45].x,points[45].y));
					AU26 = (float)Temp / Temp2;
					//if (AU26 < AU26_low && AU26 != 0f) { AU26_low = AU26;}
					//if (AU26 > AU26_high) { AU26_high = AU26;}
					//AU26 = (float)Math.Round(Mathf.InverseLerp(AU26_low,AU26_high,AU26),1);					
					AU26 = (float)Math.Round(Mathf.InverseLerp(0.1f,1f,AU26),1);					
					expressionPlayer.jawOpen_Close = (float)Math.Round(Mathf.Lerp(expressionPlayer.jawOpen_Close,AU26,AU26_lowPassFilter),1);	
					if (AU26 > 0.1) { AU25 = 1; }


					// =====================================================================================
					// AU23 - Lip Tightener (mouthNarrow_Pucker)
					
					//AU23 = (float)(Math.Atan2(points[49].y - points[50].y, points[50].x - points[49].x) + Math.Atan2(points[53].y - points[52].y,points[53].x - points[52].x) + Math.Atan2(points[61].y - points[49].y,points[61].x - points[49].x) + Math.Atan2(points[63].y - points[53].y,points[53].x - points[63].x) + Math.Atan2(points[58].y - points[59].y,points[58].x - points[59].x) + Math.Atan2(points[56].y - points[55].y,points[55].x - points[56].x) + Math.Atan2(points[60].y - points[51].y,points[51].x - points[60].x) + Math.Atan2(points[64].y - points[51].y,points[64].x - points[51].x) + Math.Atan2(points[57].y - points[60].y,points[57].x - points[60].x) + Math.Atan2(points[57].y - points[64].y,points[64].x - points[57].x) + Math.Atan2(points[62].y - points[49].y,points[62].x - points[49].x) + Math.Atan2(points[62].y - points[53].y,points[53].x - points[62].x) + Math.Atan2(points[57].y - points[60].y,points[57].x - points[60].x) + Math.Atan2(points[57].y - points[64].y,points[64].x - points[57].x));
					
					
					
					
					
					
					AU23 = (float)(Math.Atan2(points[49].y - points[50].y, points[50].x - points[49].x) + Math.Atan2(points[53].y - points[52].y,points[53].x - points[52].x) + Math.Atan2(points[61].y - points[49].y,points[61].x - points[49].x) + Math.Atan2(points[63].y - points[53].y,points[53].x - points[63].x) + Math.Atan2(points[58].y - points[59].y,points[58].x - points[59].x) + Math.Atan2(points[56].y - points[55].y,points[55].x - points[56].x) + Math.Atan2(points[60].y - points[51].y,points[51].x - points[60].x) + Math.Atan2(points[64].y - points[51].y,points[64].x - points[51].x) + Math.Atan2(points[57].y - points[60].y,points[57].x - points[60].x) + Math.Atan2(points[57].y - points[64].y,points[64].x - points[57].x) + Math.Atan2(points[62].y - points[49].y,points[62].x - points[49].x) + Math.Atan2(points[62].y - points[53].y,points[53].x - points[62].x) + Math.Atan2(points[57].y - points[60].y,points[57].x - points[60].x) + Math.Atan2(points[57].y - points[64].y,points[64].x - points[57].x));
					
					//AU23 = (float)(Math.Atan2(points[49].y - points[50].y, points[50].x - points[49].x) + Math.Atan2(points[53].y - points[52].y,points[53].x - points[52].x) +					
					//Math.Atan2(points[61].y - points[49].y,points[61].x - points[49].x) + Math.Atan2(points[63].y - points[53].y,points[53].x - points[63].x) + 
					//Math.Atan2(points[58].y - points[59].y,points[58].x - points[59].x) + Math.Atan2(points[56].y - points[55].y,points[55].x - points[56].x) + 
					//Math.Atan2(points[60].y - points[51].y,points[51].x - points[60].x) + Math.Atan2(points[64].y - points[51].y,points[64].x - points[51].x) + 
					//Math.Atan2(points[57].y - points[60].y,points[57].x - points[60].x) + Math.Atan2(points[57].y - points[64].y,points[64].x - points[57].x) + 
					//Math.Atan2(points[62].y - points[49].y,points[62].x - points[49].x) + Math.Atan2(points[62].y - points[53].y,points[53].x - points[62].x) + 
					//Math.Atan2(points[57].y - points[60].y,points[57].x - points[60].x) + Math.Atan2(points[57].y - points[64].y,points[64].x - points[57].x));


					//AU23 = (float)(Math.Atan2(points[50].y - points[49].y, points[50].x - points[49].x));
					//AU23 += (float)(Math.Atan2(points[52].y - points[53].y,points[53].x - points[52].x));					
					//AU23 += (float)(Math.Atan2(points[49].y - points[61].y,points[61].x - points[49].x));
					//AU23 += (float)(Math.Atan2(points[63].y - points[53].y,points[53].x - points[63].x)); 
					//AU23 += (float)(Math.Atan2(points[59].y - points[58].y,points[58].x - points[59].x));
					//AU23 += (float)(Math.Atan2(points[55].y - points[56].y,points[55].x - points[56].x)); 
					//AU23 += (float)(Math.Atan2(points[51].y - points[60].y,points[51].x - points[60].x)); 
					//AU23 += (float)(Math.Atan2(points[51].y - points[64].y,points[64].x - points[51].x)); //small 
					//AU23 += (float)(Math.Atan2(points[60].y - points[57].y,points[57].x - points[60].x));					
					//AU23 += (float)(Math.Atan2(points[64].y - points[57].y,points[64].x - points[57].x));
					//AU23 += (float)(Math.Atan2(points[49].y - points[62].y,points[62].x - points[49].x));  
					//AU23 += (float)(Math.Atan2(points[53].y - points[62].y,points[53].x - points[62].x)); 
					//AU23 += (float)(Math.Atan2(points[60].y - points[57].y,points[57].x - points[60].x));
					//AU23 += (float)(Math.Atan2(points[64].y - points[57].y,points[64].x - points[57].x));

					
					if (AU23 < AU23_low && AU23 != 0f) { AU23_low = AU23;}
					if (AU23 > AU23_high && AU23 != 0f) { AU23_high = AU23;}					
					AU23 = (float)Math.Round(Mathf.InverseLerp(AU23_low,AU23_high,AU23),2);	
		
					expressionPlayer.mouthNarrow_Pucker = (float)(AU23);	
					//Temp = (float)(-1 + Math.Round((AU23 * 2),2));
					//expressionPlayer.mouthNarrow_Pucker = (float)Math.Round(Mathf.Lerp(expressionPlayer.mouthNarrow_Pucker,Temp,AU23_lowPassFilter),1);								

					// =====================================================================================
					// AU7 - Eyes
					AU7_r = (distance(points[37].x,points[37].y,points[41].x,points[41].y) + distance(points[38].x,points[38].y,points[40].x,points[40].y)) / (2 * distance(points[36].x,points[36].y,points[39].x,points[39].y));
					AU7_l = (distance(points[43].x,points[43].y,points[47].x,points[47].y) + distance(points[44].x,points[44].y,points[46].x,points[46].y)) / (2 * distance(points[42].x,points[42].y,points[45].x,points[45].y));

					if (AU7_r < AU7_r_low && AU7_r != 0f) { AU7_r_low = AU7_r;}
					if (AU7_r > AU7_r_high) { AU7_r_high = AU7_r;}
					AU7_r = (float)Math.Round(Mathf.InverseLerp(AU7_r_low,(AU7_r_high - 0.01f),AU7_r),1);					

					if (AU7_l < AU7_l_low && AU7_l != 0f) { AU7_l_low = AU7_l;}
					if (AU7_l > AU7_l_high) { AU7_l_high = AU7_l;}
					AU7_l = (float)Math.Round(Mathf.InverseLerp(AU7_l_low,(AU7_l_high - 0.01f),AU7_l),1);					
					AU7 = (float)((AU7_r + AU7_l)/2); 

					Temp = (float)(-1f + (AU7_r * 2));			
					expressionPlayer.rightEyeOpen_Close = (float)Math.Round(Mathf.Lerp(expressionPlayer.rightEyeOpen_Close,Temp,AU7_lowPassFilter),1);
					Temp = (float)(-1f + (AU7_l * 2));
					expressionPlayer.leftEyeOpen_Close = (float)Math.Round(Mathf.Lerp(expressionPlayer.rightEyeOpen_Close,Temp,AU7_lowPassFilter),1);					
					
					
					// =====================================================================================
					// AU9 - Nose Wrinkler
					// PYTHON AU9 = (math.atan2(y[50] - y[33], x[33] - x[50]) + math.atan2(y[52] - y[33],x[52] - x[33]))
					AU9 = (float)(Math.Atan2(points[50].y - points[33].y, points[33].x - points[50].x) + Math.Atan2(points[52].y - points[33].y,points[52].x - points[33].x));
					AU9 = Mathf.Abs(AU9);

					if (AU9 < AU9_low && AU9 != 0f) { AU9_low = AU9;}
					if (AU9 > AU9_high) { AU9_high = AU9;}
					AU9 = (float)Math.Round(Mathf.InverseLerp(AU9_low,AU9_high,AU9),2);
					//expressionPlayer.noseSneer = (float)(1f - AU9);
					Temp = (float)(1f - AU9);
					expressionPlayer.noseSneer = (float)Math.Round(Mathf.Lerp(expressionPlayer.noseSneer,Temp,AU9_lowPassFilter),2);
					
					// =====================================================================================

					//Debug.Log("counter:" + counter + " AU1=" + (AU1_r + AU1_l));
					//Debug.Log("counter:" + counter + " AU2_r=" + AU2_r + " AU2_l=" + AU2_l);
					//Debug.Log("counter:" + counter + " AU4=" + AU4 + " low=" + AU4_low + "  high=" + AU4_high);
					//Debug.Log("counter:" + counter + " AU6=" + AU6_l + "   " + AU6_r);

					//Debug.Log("counter:" + counter + " AU26=" + AU26);
					Debug.Log("counter:" + counter + " AU23=" + AU23 + "   - " + AU23_low + "/" + AU23_high + "        ==" + expressionPlayer.mouthNarrow_Pucker);										
					
					//Debug.Log("counter:" + counter + " AU7_r=" + AU7_r + " low=" + AU7_r_low + "  high=" + AU7_r_high);		

					//Debug.Log("counter:" + counter + " AU9=" + AU9 + " low=" + AU9_low + "  high=" + AU9_high);


				}
			}	
		
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

		protected virtual float lerp(float x, float a, float b)	
		{
			float ret = (x - a) / (b - a);
			if (ret < 0) { ret = 0; };
			return (float)Math.Round(ret,2);		
		}		

        private void DrawFaceLandmark(Mat imgMat, List<Point> points, Scalar color, int thickness, bool drawIndexNumbers = false)
        {
            if (points.Count == 5)
            {

                Imgproc.line(imgMat, points[0], points[1], color, thickness);
                Imgproc.line(imgMat, points[1], points[4], color, thickness);
                Imgproc.line(imgMat, points[4], points[3], color, thickness);
                Imgproc.line(imgMat, points[3], points[2], color, thickness);

            }
            else if (points.Count == 68)
            {

                for (int i = 1; i <= 16; ++i)
                    Imgproc.line(imgMat, points[i], points[i - 1], color, thickness);

                for (int i = 28; i <= 30; ++i)
                    Imgproc.line(imgMat, points[i], points[i - 1], color, thickness);

                for (int i = 18; i <= 21; ++i)
                    Imgproc.line(imgMat, points[i], points[i - 1], color, thickness);
                for (int i = 23; i <= 26; ++i)
                    Imgproc.line(imgMat, points[i], points[i - 1], color, thickness);
                for (int i = 31; i <= 35; ++i)
                    Imgproc.line(imgMat, points[i], points[i - 1], color, thickness);
                Imgproc.line(imgMat, points[30], points[35], color, thickness);

                for (int i = 37; i <= 41; ++i)
                    Imgproc.line(imgMat, points[i], points[i - 1], color, thickness);
                Imgproc.line(imgMat, points[36], points[41], color, thickness);

                for (int i = 43; i <= 47; ++i)
                    Imgproc.line(imgMat, points[i], points[i - 1], color, thickness);
                Imgproc.line(imgMat, points[42], points[47], color, thickness);

                for (int i = 49; i <= 59; ++i)
                    Imgproc.line(imgMat, points[i], points[i - 1], color, thickness);
                Imgproc.line(imgMat, points[48], points[59], color, thickness);

                for (int i = 61; i <= 67; ++i)
                    Imgproc.line(imgMat, points[i], points[i - 1], color, thickness);
                Imgproc.line(imgMat, points[60], points[67], color, thickness);
            }
            else
            {
                for (int i = 0; i < points.Count; i++)
                {
                    Imgproc.circle(imgMat, points[i], 2, color, -1);
                }
            }

            // Draw the index number of facelandmark points.
            if (drawIndexNumbers)
            {
                for (int i = 0; i < points.Count; ++i)
                    Imgproc.putText(imgMat, i.ToString(), points[i], Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(255, 255, 255, 255), 1, Imgproc.LINE_AA, false);
            }
        }	
				
#if (UNITY_IOS && UNITY_2018_1_OR_NEWER) || (UNITY_ANDROID && UNITY_2018_3_OR_NEWER)
        bool isUserRequestingPermission;

        IEnumerator OnApplicationFocus(bool hasFocus)
        {
            yield return null;

            if (isUserRequestingPermission && hasFocus)
                isUserRequestingPermission = false;
        }
#endif

        void OnDestroy()
        {

            //if (faceLandmarkDetector != null)
            //    faceLandmarkDetector.Dispose();
		
#if UNITY_WEBGL && !UNITY_EDITOR
            if (getFilePath_Coroutine != null)
            {
                StopCoroutine(getFilePath_Coroutine);
                ((IDisposable)getFilePath_Coroutine).Dispose();
            }
#endif
        }


    }

}

#endif