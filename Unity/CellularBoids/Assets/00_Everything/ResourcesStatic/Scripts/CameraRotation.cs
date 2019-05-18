using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraRotation : MonoBehaviour
{
    float radius = 1f;

    void Start()
    {
        radius = transform.position.magnitude;    
    }

    // Update is called once per frame
    void Update()
    {
        float angle = Time.time*0.2f;
        transform.position = new Vector3( radius * Mathf.Cos(angle), 0f, radius *Mathf.Sin(angle));

        transform.LookAt(Vector3.zero);

    }
}
