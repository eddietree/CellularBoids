using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine.Jobs;

public class BoidsManager : MonoBehaviour
{
    const int NumCells = 512;
    public GameObject PfbCell; // prototype

    Transform [] _cellTfms;
    TransformAccessArray _cellTfmAccessArray;
    GameObject [] _cellObjs;
    NativeArray<Vector3> _cellVelocities;

    PositionUpdateJob _jobPos;
    JobHandle _jobHandlePosition;
    //JobHandle m_AccelJobHandle;

    struct PositionUpdateJob : IJobParallelForTransform
    {
        [ReadOnly]
        public NativeArray<Vector3> velocity;  // the velocities from AccelerationJob

        public float deltaTime;

        public void Execute(int i, TransformAccess transform)
        {
            transform.position += velocity[i] * deltaTime;
        }
    }

    void Start()
    {
        Setup();
    }

    void Update()
    {
        _jobPos = new PositionUpdateJob()
        {
            deltaTime = Time.deltaTime,
            velocity = _cellVelocities,
        };

        //m_AccelJobHandle = m_AccelJob.Schedule(m_ObjectCount, 64);
        _jobHandlePosition = _jobPos.Schedule(_cellTfmAccessArray);
    }

    public void LateUpdate()
    {
        _jobHandlePosition.Complete();
    }

    private void OnDestroy()
    {
        _cellVelocities.Dispose();
        _cellTfmAccessArray.Dispose();
    }

    void Setup()
    {
        _cellObjs = new GameObject[NumCells];
        _cellTfms = new Transform[NumCells];
        _cellVelocities = new NativeArray<Vector3>(NumCells, Allocator.Persistent);

        float radius = 10f;

        for( int i = 0; i < NumCells; ++i)
        {
            var newCell = GameObject.Instantiate(PfbCell);
            newCell.transform.position = new Vector3( UnityEngine.Random.Range(-radius, radius), UnityEngine.Random.Range(-radius, radius), UnityEngine.Random.Range(-radius, radius));

            _cellObjs[i] = newCell;
            _cellTfms[i] = newCell.transform;

            _cellVelocities[i] = Vector3.up;
        }

        _cellTfmAccessArray = new TransformAccessArray(_cellTfms);
    }
}
