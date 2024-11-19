// Copyright (c) 2021 homuler
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

// ATTENTION!: This code is for a tutorial.

using System;
using System.Threading.Tasks;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Mediapipe.Unity.CoordinateSystem;
using Mediapipe.Tasks.Vision.FaceLandmarker;

using OpenCVForUnity.UnityUtils.Helper;
using OpenCVForUnity.UnityUtils;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.FaceModule;

using OpenCVForUnity.UtilsModule;
using OpenCVForUnityExample;
using OpenCVForUnity.VideoioModule;

using Stopwatch = System.Diagnostics.Stopwatch;

namespace Mediapipe.Unity.Tutorial
{
  public class FaceMesh : MonoBehaviour
  {
	[SerializeField] private TextAsset _configAsset;  
    [SerializeField] private RawImage _screen;
    [SerializeField] private int _width;
    [SerializeField] private int _height;
    [SerializeField] private int _fps;

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
	private UnityEngine.Rect screenRect;
	
	int requestedMode = 0;
	bool video_capture_initialised = false;
	const int K_Webcam = 0;
	const int K_Video = 1;

	VideoCapture capture;
	bool isPlaying = false;
	bool shouldUpdateVideoFrame = false;
	long prevFrameTickCount;
	long currentFrameTickCount;	
	//static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/01-01-05-01-01-01-01_1.mp4";
	static readonly string VIDEO_FILENAME = "OpenCVForUnity/emotion_videos/RAVDESS_320_Labelled/anger/Video_Man01.mp4";
	Mat rgbVideoMat;
	Texture2D Video_texture;

	VideoWriter writer;	
	Mat recordingFrameRgbMat;	
	const int numFinalLandmarkPoints = 68;
	
    private IEnumerator Start()
    {
		List<Vector2> points;
		
		// Set Request Mode (K_Webcam or K_Video)
		requestedMode = K_Video;
		points = new List<Vector2>();
		
		if (requestedMode == K_Webcam)
		{		
			if (WebCamTexture.devices.Length == 0)
			{
				throw new System.Exception("Web Camera devices are not found");
			}

			var webCamDevice = WebCamTexture.devices[0];
			_webCamTexture = new WebCamTexture(webCamDevice.name, _width, _height, _fps);
			_webCamTexture.Play();
			yield return new WaitUntil(() => _webCamTexture.width > 16);	
		
		} else {
			
			/*
			rgbVideoMat = new Mat();
			
			capture = new VideoCapture();
			capture.open(OpenCVForUnity.UnityUtils.Utils.getFilePath(VIDEO_FILENAME));
			video_capture_initialised = true;
			
			capture.grab();
			capture.retrieve(rgbVideoMat);
			Video_texture = new Texture2D(rgbVideoMat.cols(), rgbVideoMat.rows(), TextureFormat.RGB24, false);
		
			//gameObject.GetComponent<Renderer>().material.mainTexture = Video_texture;
			
			StartCoroutine("WaitFrameTime");
			capture.set(Videoio.CAP_PROP_POS_FRAMES, 0);
			isPlaying = true;			
			*/
			
		}

		// ==============================================================================================================
		// Setup Video Output
		writer = new VideoWriter();
		//writer.open(Application.persistentDataPath + "/output.mp4", Videoio.CAP_OPENCV_MJPEG, VideoWriter.fourcc('M', 'J', 'P', 'G'), 30, new Size((int)_width, (int)_height));
		writer.open(Application.persistentDataPath + "/output.mp4", VideoWriter.fourcc('X', '2', '6', '4'), 30, new Size((int)_width, (int)_height));
		recordingFrameRgbMat = new Mat((int)_height, (int)_width, CvType.CV_8UC3);
		Debug.Log("File : " + Application.persistentDataPath + "/output.mp4");
		
		// C:\Users\Darren Bellenger\AppData\LocalLow\DefaultCompany\MediaPipeUnityPlugin
		
		if (!writer.isOpened())
		{
			Debug.LogError("writer.isOpened() false");
			//writer.release();
		}

		
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
			
			// Update input image
			if (requestedMode == K_Webcam)
			{		

				_inputTexture.SetPixels32(_webCamTexture.GetPixels32(_inputPixelData));
				currentFrame = 1;
				previousFrame = 0;
			
			} else {
				
				ProcessVideo();

			}	

			
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
					//if (outputVideo.TryReadPixelData(_outputPixelData))
					//{
						//_outputTexture.SetPixels32(_outputPixelData);
						//_outputTexture.Apply();
					//}
				}

				var multiFaceLandmarksPacket = task2.Result.packet;
				if (multiFaceLandmarksPacket != null)
				{
					var multiFaceLandmarks = multiFaceLandmarksPacket.Get(NormalizedLandmarkList.Parser);
					if (multiFaceLandmarks != null && multiFaceLandmarks.Count > 0)
					{
						foreach (var landmarks in multiFaceLandmarks)
						{
							// top of the head
							var topOfHead = landmarks.Landmark[10];
							//Debug.Log($"Unity Local Coordinates: {screenRect.GetPoint(topOfHead)}, Image Coordinates: {topOfHead}");
							
							points.Clear();
							points = ConvertMediaPipeLandmarksToVectorList(landmarks);
							if ( points.Count == 68) { 

								_outputTexture.SetPixels(_inputTexture.GetPixels());			
								for (int i = 0; i < numFinalLandmarkPoints; i++)
								{
									_outputTexture.SetPixel( (int)points[i].x, (int)(_height - points[i].y), UnityEngine.Color.white );
								}					
								_outputTexture.Apply();
						
							
							}
							
						}
					}
				}
				
				Utils.texture2DToMat(_outputTexture, recordingFrameRgbMat);
				Imgproc.cvtColor(recordingFrameRgbMat, recordingFrameRgbMat, Imgproc.COLOR_RGB2BGR);
				writer.write(recordingFrameRgbMat);
				
			}
			
			previousFrame = currentFrame;
		}
	  
    }
	
	private List<Vector2> ConvertMediaPipeLandmarksToVectorList(Mediapipe.NormalizedLandmarkList mp_pts, List<Vector2> pts = null)
	{
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

		Vector3 temp; 
		
		if (pts == null)
		{
			pts = new List<Vector2>();
		}
		pts.Clear();

		for (int i = 0; i < numFinalLandmarkPoints; i++)
		{
			//temp = screenRect.GetPoint(mp_pts.Landmark[landmarks_68[i]]); //_pipeline.GetFaceVertex( landmarks_68[i]).xy;
			float x = (float)Math.Round(((mp_pts.Landmark[alt_landmarks_68[i]].X) * _width),2);
			float y = (float)Math.Round(((mp_pts.Landmark[alt_landmarks_68[i]].Y) * _height),2);
			pts.Add(new Vector2(x, y));
			//Debug.Log("Points:" + x + "/" + y);
			
			//pts.Add(new Vector2(temp.x, temp.y));
			//Debug.Log("temp:" + temp + "    X: " + mp_pts.Landmark[10].X + "    Landmark: " + mp_pts.Landmark[10]);
			
		}		
		
		return pts;
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

    void Update () {

		//StartCoroutine("ProcessImage");
		//counter += 1;
		//Debug.Log("count=" + counter);

		/*
		if (requestedMode == K_Video)
		{
			ProcessVideo();
		}
		*/
		
	}
	
	void ProcessVideo ()	
	{

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
		
			//Loop play
			if ((capture.get(Videoio.CAP_PROP_POS_FRAMES) + 1) >= capture.get(Videoio.CAP_PROP_FRAME_COUNT))
			{
				capture.set(Videoio.CAP_PROP_POS_FRAMES, 0);
				Debug.Log("Loop video");
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

	/*
    private IEnumerator ProcessImage () {

		_inputTexture.SetPixels32(_webCamTexture.GetPixels32(_inputPixelData));
		var imageFrame = new ImageFrame(ImageFormat.Types.Format.Srgba, _width, _height, _width * 4, _inputTexture.GetRawTextureData<byte>());
		var currentTimestamp = stopwatch.ElapsedTicks / (System.TimeSpan.TicksPerMillisecond / 1000);
		_graph.AddPacketToInputStream("input_video", Packet.CreateImageFrameAt(imageFrame, currentTimestamp));

		//var task = _multiFaceLandmarksStream.WaitNextAsync();			
        //yield return new WaitUntil(() => task.IsCompleted);
        //var result = task.Result;

        var task1 = _outputVideoStream.WaitNextAsync();
        var task2 = _multiFaceLandmarksStream.WaitNextAsync();
        var task = Task.WhenAll(task1, task2);
		
        //if (!result.ok)
        //{
        //  throw new Exception("Something went wrong");
        //}	

        yield return new WaitUntil(() => task.IsCompleted);

        if (!task1.Result.ok || !task2.Result.ok)
        {
          throw new System.Exception("Something went wrong");
        }		
	
        var multiFaceLandmarksPacket = task2.Result.packet;
        if (multiFaceLandmarksPacket != null)
        {
			Debug.Log("Landmarks:" + multiFaceLandmarksPacket);	
            var multiFaceLandmarks = multiFaceLandmarksPacket.Get(NormalizedLandmarkList.Parser);
          //_multiFaceLandmarksAnnotationController.DrawNow(multiFaceLandmarks);
			if (multiFaceLandmarks != null && multiFaceLandmarks.Count > 0)
			{
				foreach (var landmarks in multiFaceLandmarks)
				{
					// top of the head
					var topOfHead = landmarks.Landmark[10];
					Debug.Log($"Unity Local Coordinates: {screenRect.GetPoint(topOfHead)}, Image Coordinates: {topOfHead}");
				}
			}		  
        }
        else
        {
          //_multiFaceLandmarksAnnotationController.DrawNow(null);
        }
		
		
		Debug.Log("end of landmark prediction");	
	
	}	
	*/
	
    private void OnDestroy()
    {
		
		//writer.close();
		writer.release();			
		
		Debug.Log("OnDestroy");	

		if (_webCamTexture != null)
		{
			_webCamTexture.Stop();
		}
	  
		_outputVideoStream?.Dispose();
		_outputVideoStream = null;
		_multiFaceLandmarksStream?.Dispose();
		_multiFaceLandmarksStream = null;

      if (_graph != null)
      {
        try
        {
          _graph.CloseInputStream("input_video");
          _graph.WaitUntilDone();
        }
        finally
        {
          _graph.Dispose();
          _graph = null;
        }
      }
    }
  }
}
