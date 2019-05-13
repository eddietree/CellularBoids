using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine.Jobs;

public class BoidsManager : MonoBehaviour
{
    const int NumCells = 4096;
    const int NumGroups = 8;

    public GameObject PfbCell; // prototype

    Transform [] _cellTfms;
    TransformAccessArray _cellTfmAccessArray;
    GameObject [] _cellObjs;
    MeshRenderer[] _cellRenderers;
    MaterialPropertyBlock[] _cellMatProperyBlock;

    NativeArray<int> _cellGroupIndex;
    NativeArray<Vector3> _cellPositions;
    NativeArray<Vector3> _cellVelocities;
    NativeArray<float> _cellGroupsForceMatrix;

    PositionUpdateJob _jobPos;
    CellularForceJob _jobCellularForce;
    JobHandle _jobHandlePosition;
    JobHandle _jobsHandleCellularForce;

    struct CellularForceJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<Vector3> position;
        public NativeArray<Vector3> velocity; 

        [ReadOnly] public NativeArray<int> groupIndex;
        [ReadOnly] public NativeArray<float> forceMatrix;
        [ReadOnly] public int numCells;

        public float deltaTime;

        public void Execute(int i)
        {
            // apply force from all
            Vector3 currVel = velocity[i];

            currVel.x = 1f;
            currVel.y = 0;
            currVel.z = 0f;

            velocity[i] = currVel;
        }
    }

    struct PositionUpdateJob : IJobParallelForTransform
    {
        public NativeArray<Vector3> position;  // the velocities from AccelerationJob
        [ReadOnly] public NativeArray<Vector3> velocity;  // the velocities from AccelerationJob

        public float deltaTime;

        public void Execute(int i, TransformAccess transform)
        {
            position[i] += velocity[i] * deltaTime;
            transform.position = position[i];
        }
    }

    void Start()
    {
        Setup();
    }

    void Update()
    {
        _jobCellularForce = new CellularForceJob()
        {
            numCells = NumCells,
            position = _cellPositions,
            velocity = _cellVelocities,
            groupIndex = _cellGroupIndex,
            forceMatrix = _cellGroupsForceMatrix,

            deltaTime = Time.deltaTime,
        };

        _jobPos = new PositionUpdateJob()
        {
            position = _cellPositions,
            velocity = _cellVelocities,
            deltaTime = Time.deltaTime,
        };

        _jobsHandleCellularForce = _jobCellularForce.Schedule(NumCells, 64);
        _jobHandlePosition = _jobPos.Schedule(_cellTfmAccessArray, _jobsHandleCellularForce);
    }

    public void LateUpdate()
    {
        _jobHandlePosition.Complete();
    }

    private void OnDestroy()
    {
        _cellVelocities.Dispose();
        _cellGroupIndex.Dispose();
        _cellPositions.Dispose();
        _cellGroupsForceMatrix.Dispose();
        _cellTfmAccessArray.Dispose();
    }

    void Setup()
    {
        _cellObjs = new GameObject[NumCells];
        _cellTfms = new Transform[NumCells];
        _cellRenderers = new MeshRenderer[NumCells];
        _cellMatProperyBlock = new MaterialPropertyBlock[NumCells];

        _cellGroupIndex = new NativeArray<int>(NumCells, Allocator.Persistent);
        _cellVelocities = new NativeArray<Vector3>(NumCells, Allocator.Persistent);
        _cellPositions = new NativeArray<Vector3>(NumCells, Allocator.Persistent);

        float radius = 10f;

        for( int i = 0; i < NumCells; ++i)
        {
            var newCellGroupIndex = UnityEngine.Random.Range(0, NumGroups);

            var newCell = GameObject.Instantiate(PfbCell);

            _cellPositions[i] = new Vector3( UnityEngine.Random.Range(-radius, radius), UnityEngine.Random.Range(-radius, radius), UnityEngine.Random.Range(-radius, radius));
            _cellObjs[i] = newCell;
            _cellTfms[i] = newCell.transform;
            _cellGroupIndex[i] = newCellGroupIndex;
            _cellVelocities[i] = Vector3.up;

            // rendering
            var renderer = newCell.GetComponent<MeshRenderer>();
            MaterialPropertyBlock materialBlock;
            renderer.GetPropertyBlock(materialBlock = new MaterialPropertyBlock());
            _cellRenderers[i] = renderer;
            _cellMatProperyBlock[i] = materialBlock;

            materialBlock.SetColor("_Color", Color.HSVToRGB((float)newCellGroupIndex / (float)NumGroups, 1f, 1f));
            renderer.SetPropertyBlock(materialBlock);
        }
        _cellTfmAccessArray = new TransformAccessArray(_cellTfms);

        // force matrix
        _cellGroupsForceMatrix = new NativeArray<float>(NumGroups * NumGroups, Allocator.Persistent);
        for(int i = 0; i < _cellGroupsForceMatrix.Length; ++i)
        {
            _cellGroupsForceMatrix[i] = UnityEngine.Random.Range(-1f, 1f);
        }

    }
}
