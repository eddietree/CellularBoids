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

    //
    NativeArray<float> _cellGroupsForceMatrix;

    PositionUpdateJob _jobPos;
    CellularForceJob _jobCellularForce;
    JobHandle _jobHandlePosition;
    JobHandle _jobsHandleCellularForce;

    struct CellularForceJob : IJobParallelFor
    {
        // cells
        [ReadOnly] public NativeArray<Vector3> position;
        public NativeArray<Vector3> velocity; 
        [ReadOnly] public NativeArray<int> groupIndex;

        // global
        [ReadOnly] public NativeArray<float> forceMatrix;
        [ReadOnly] public int numCells;
        [ReadOnly] public int numGroups;

        public float deltaTime;


        public void Execute(int i)
        {
            // apply force from all
            Vector3 currVel = velocity[i];
            Vector3 currPos = position[i];
            int currGroupIndex = groupIndex[i];

            const float forceCoeffAttract = 0.2f;
            const float mass = 0.5f;
            float radiusMin = 0.2f; // TODO!!! use ball radius
            float radiusMax = 10f; // TODO!!! use ball radius
            float radiusMinSqr = radiusMin * radiusMin;
            float radiusMaxSqr = radiusMax * radiusMax;

            Vector3 forceAccum = Vector3.zero;

            // TODO: optimize only the cells nearby
            for (int j = 0; j < NumCells; ++j)
            {
                if (i == j)
                    continue;

                Vector3 otherPos = position[j];
                int otherGroupIndex = groupIndex[j];

                Vector3 dirToOtherPos = otherPos - currPos;
                Vector3 dirToOtherPosNorm = dirToOtherPos.normalized;
                float distToOtherPosSqr = Vector3.Dot(dirToOtherPos, dirToOtherPos);

                // repulsion
                forceAccum -= dirToOtherPosNorm * Mathf.Exp(-30f * distToOtherPosSqr);

                // cellular attraction?
                float cellularForce = forceMatrix[currGroupIndex * numGroups + otherGroupIndex];
                forceAccum += cellularForce * dirToOtherPosNorm * Mathf.Exp(-1f * distToOtherPosSqr) * forceCoeffAttract;
            }

            // dist from edge
            float edgeRadius = 4f;
            float distFromEdge = Mathf.Max(0,edgeRadius - currPos.magnitude);
            forceAccum += -currPos.normalized * Mathf.Exp(-5f * distFromEdge);

            Vector3 accel = forceAccum / mass;

            currVel += accel * deltaTime;

            const float drag = 2f;
            currVel = currVel * (1f - deltaTime * drag);
            //currVel *= 0.95f;

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
            transform.rotation = Quaternion.LookRotation(velocity[i]);
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
            numGroups = NumGroups,
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

        float radius = 2f;

        for( int i = 0; i < NumCells; ++i)
        {
            var newCellGroupIndex = UnityEngine.Random.Range(0, NumGroups);

            var newCell = GameObject.Instantiate(PfbCell);

            _cellPositions[i] = new Vector3( UnityEngine.Random.Range(-radius, radius), UnityEngine.Random.Range(-radius, radius), UnityEngine.Random.Range(-radius, radius));
            _cellObjs[i] = newCell;
            _cellTfms[i] = newCell.transform;
            _cellGroupIndex[i] = newCellGroupIndex;
            _cellVelocities[i] = (new Vector3(UnityEngine.Random.Range(-radius, radius), UnityEngine.Random.Range(-radius, radius), UnityEngine.Random.Range(-radius, radius)))* 0.1f;

            // rendering
            var renderer = newCell.GetComponent<MeshRenderer>();
            MaterialPropertyBlock materialBlock;
            renderer.GetPropertyBlock(materialBlock = new MaterialPropertyBlock());
            _cellRenderers[i] = renderer;
            _cellMatProperyBlock[i] = materialBlock;

            var color = Color.HSVToRGB((float)newCellGroupIndex / (float)NumGroups, 1f, 1f);
            materialBlock.SetColor("_Color", color);
            renderer.SetPropertyBlock(materialBlock);


            var lineRenderer = renderer.GetComponentInChildren<TrailRenderer>();
            if (lineRenderer != null)
                lineRenderer.startColor = lineRenderer.endColor = color;
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
