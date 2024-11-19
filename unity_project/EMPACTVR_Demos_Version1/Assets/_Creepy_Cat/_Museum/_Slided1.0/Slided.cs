// Slided engine 1.1 par Creepy Cat (C)2017
// you should not resale it directly (or modified).

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Slided : MonoBehaviour {
	
	//L'interface
	public  GameObject slidepivot;

	public float camspeed = 14.5f;
	public float pivotspeed = 12.5f;

	public float decalX=0.0f;
	public float decalY=0.0f;
	public float zoomValue=0.9f;

	// Current waypoint index
	public int currentWaypoint = 0; //First Slide index
	static int totalWaypoint = 0; 	//Total Slide number
	public GameObject[] waypoints;

	// Positions du pivot
	private  int previoustWaypoint = 0; //Previous slide index

	private  float currentPivotX;
	private  float  currentPivotY;
	private  float  currentPivotZ;

	private  float  nextPivotX;
	private  float  nextPivotY;
	private  float  nextPivotZ;

	// Positions des slides
	private  float currentSlideX;
	private  float  currentSlideY;
	private  float  currentSlideZ;

	private  float  nextslideX;
	private  float  nextslideY;
	private  float  nextslideZ;

	// Variables de l'engine
	private  float cameraSmooth;
	private  float pivotSmooth;

	private  float oldDecalX ;
	private  float oldDecalY ;

	// Wake up neo
	void Awake () {

		oldDecalX = decalX;
		oldDecalY = decalY;

		// If more than 0 waypoint
		if (waypoints.Length > 0) { 

			// getting first slide coordinates
			Vector3 posi =  waypoints[0].transform.TransformPoint (new Vector3(decalX,decalY,zoomValue));

			nextslideX=posi.x;
			nextslideY=posi.y;
			nextslideZ=posi.z;


			currentSlideX=posi.x;
			currentSlideY=posi.y;
			currentSlideZ=posi.z;

			nextPivotX=nextslideX;
			nextPivotY=nextslideY;
			nextPivotZ=nextslideZ;

			currentPivotX=nextslideX;
			currentPivotY=nextslideY;
			currentPivotZ=nextslideZ;

			currentPivotX=waypoints[currentWaypoint].transform.position.x;
			currentPivotY=waypoints[currentWaypoint].transform.position.y;
			currentPivotZ=waypoints[currentWaypoint].transform.position.z;

		}

		oldDecalX = decalX;
		oldDecalY = decalY;

	}
	
	// Update is called once per frame
	void Update () {
		GetInputKey();
		FacingCamera();
	}

	// -----------------------------------------
	// Usefull functions to interact with slided
	// -----------------------------------------
	void OnTriggerEnter(Collider other) {
		Debug.Log("Slide enter " + currentWaypoint);
	}

	void OnTriggerExit(Collider other) {
		Debug.Log("Slide exit " + previoustWaypoint);
	}

	public void Reset3DView() {
		decalX=0.0f;
		decalY=0.0f;
	}	

	public void Set3DView() {
		decalX=oldDecalX;
		decalY=oldDecalY;
	}

	public void GoToNextSlide() {
		previoustWaypoint = currentWaypoint;
		currentWaypoint++; 

		if (currentWaypoint>= waypoints.Length ){
			currentWaypoint= waypoints.Length-1;
		}	
	}

	public void GoToPrevSlide() {
		previoustWaypoint = currentWaypoint;
		currentWaypoint--; 

		if (currentWaypoint<0 ){
			currentWaypoint= 0;
		}	
	}

	public void GoToSlideNumber(int num) {
		previoustWaypoint = currentWaypoint;
		currentWaypoint = num; 

		if (currentWaypoint<0 ){
			currentWaypoint= 0;
		}	

		if (currentWaypoint>= waypoints.Length ){
			currentWaypoint= waypoints.Length-1;
		}
	}


	//-----------------------------------------
	// Deplacement de la camera avec pivot
	// ----------------------------------------
	private void FacingCamera(){

		// Move camera and pivot
		cameraSmooth=Time.deltaTime * camspeed;
		pivotSmooth=Time.deltaTime * pivotspeed;

		// New pivot position searching
		nextPivotX=waypoints[currentWaypoint].transform.position.x;
		nextPivotY=waypoints[currentWaypoint].transform.position.y;
		nextPivotZ=waypoints[currentWaypoint].transform.position.z;

		Vector3 posi =  waypoints[currentWaypoint].transform.TransformPoint (new Vector3(decalX,decalY,zoomValue));

		nextslideX=posi.x;
		nextslideY=posi.y;
		nextslideZ=posi.z;


		// Calculate the orientation and other shit : pivot+camera
		currentPivotX=InterpolateValue(currentPivotX, nextPivotX,pivotSmooth,0.0f);
		currentPivotY=InterpolateValue(currentPivotY, nextPivotY,pivotSmooth,0.0f);
		currentPivotZ=InterpolateValue(currentPivotZ, nextPivotZ,pivotSmooth,0.0f);

		slidepivot.transform.position = new Vector3(currentPivotX,currentPivotY, currentPivotZ);

		currentSlideX=InterpolateValue(currentSlideX, nextslideX,cameraSmooth,0.0f);
		currentSlideY=InterpolateValue(currentSlideY, nextslideY,cameraSmooth,0.0f);
		currentSlideZ=InterpolateValue(currentSlideZ, nextslideZ,cameraSmooth,0.0f);

		GetComponent<Camera>().transform.position = new Vector3(currentSlideX, currentSlideY, currentSlideZ);

		// Pointing camera to pivot
		GetComponent<Camera>().transform.LookAt(slidepivot.transform); 
	}

	private void GetInputKey(){

		//Right key
		if (Input.GetKeyDown ("right") ){
			GoToNextSlide();
		}	

		//Left Key
		if (  Input.GetKeyDown ("left") ){
			GoToPrevSlide();
		}

	}

	//----------------------------------------------------------------
	// Interpolate a value to another with smooth (smooth must ba < 1)
	// ---------------------------------------------------------------
	public float InterpolateValue(float ValueA,float ValueB,float Smooth,float Velocity) {
		return  Mathf.SmoothDamp(ValueA,ValueB,ref Velocity,Smooth);		
	}

	//---------------------
	// Interpolate angle
	// --------------------
	public float InterpolateAngle(float ValueA,float ValueB,float Smooth) {
		float ix=Mathf.Sin(ValueA);
		float iy=Mathf.Cos(ValueA);		
		float jx=Mathf.Sin(ValueB);
		float jy=Mathf.Cos(ValueB);

		return Mathf.Atan2(ix-(ix-jx)*Smooth,iy-(iy-jy)*Smooth);
	}
}
