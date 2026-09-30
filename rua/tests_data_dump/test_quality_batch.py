import json

import pytest

from data_dump import quality_pipeline, training_dump
from test_quality_pipeline import write_oracle
from test_training_dump import make_episode


def replay_fixture(source, output):
    rows = [{
        'task_step': i, 'goal_predicates': [{'objects': ['bowl', 'basket_region'],
                                          'satisfied': i == 19}],
        'eef_position_m': [0.2, 0, 0.1], 'gripper_command': 1,
        'grasped_objects': ['bowl'] if i >= 10 else [],
        'object_positions_m': {'bowl': [0, 0, 0.04]},
        'all_object_positions_m': {'bowl': [0, 0, 0.04]},
    } for i in range(20)]
    write_oracle(source, output, rows)
    report = json.loads((output / 'oracle-manifest.json').read_text())
    report['object_horizontal_radius_m'] = {'bowl': 0.04}
    return report


def test_mixed_batch_keeps_good_data_and_accounts_for_every_source(tmp_path, monkeypatch):
    pytest.importorskip('lerobot.datasets.lerobot_dataset')
    sources = {name: make_episode(tmp_path, name, task_steps=20) for name in
               ('good', 'missing', 'corrupt', 'replay', 'classification', 'camera')}
    (sources['missing'] / 'environment-steps.jsonl').unlink()
    (sources['corrupt'] / 'result.json').write_text('{broken')
    original_classify = quality_pipeline.classify
    original_shape = training_dump.image_shape

    def replay(source, output):
        if source.name == 'replay':
            raise ValueError('Replay diverged at attempt 2')
        return replay_fixture(source, output)

    def classify(source, output):
        if source.name == 'classification':
            raise ValueError('Oracle feedback does not match source')
        return original_classify(source, output)

    def shape(source, camera='front'):
        if source.name == 'camera' and camera == 'wrist':
            return (32, 32, 3)
        return original_shape(source, camera)

    monkeypatch.setattr(quality_pipeline, 'replay', replay)
    monkeypatch.setattr(quality_pipeline, 'classify', classify)
    monkeypatch.setattr(training_dump, 'image_shape', shape)
    output = tmp_path / 'output'
    summary = quality_pipeline.run(list(sources.values()), output)
    assert summary['sources'] == 6
    assert summary['preprocessing_failed_sources'] == 4
    assert summary['rejected_sources'] == 5
    assert summary['exported_episodes'] == 1
    feedback = [json.loads(line) for line in (output / 'dump/feedback.jsonl').read_text().splitlines()]
    assert len(feedback) == 6
    assert {row['rejection_code'] for row in feedback} == {
        None, 'source_invalid', 'replay_failed', 'classification_failed', 'camera_incompatible'}
    for row in feedback:
        if row['rejection_code']:
            assert row['wla_training_starts'] == row['quality_nominal_starts'] == row['quality_recovery_starts'] == 0
    camera = next(row for row in feedback if row['source'] == str(sources['camera']))
    assert camera['quality_role_steps']['nominal'] == 20
    starts = (output / 'dump/rua_lerobot/meta/quality-starts.jsonl').read_text().splitlines()
    assert len(starts) == summary['training_starts'] == 12
    trace = [json.loads(line) for line in (output / 'dump/trace.jsonl').read_text().splitlines()]
    assert any(row['source'] == str(sources['corrupt']) and row['kind'] == 'environment' for row in trace)


def test_all_input_failures_write_zero_manifest_without_dataset(tmp_path):
    source = make_episode(tmp_path, 'broken', task_steps=20)
    (source / 'result.json').write_text('[]')
    output = tmp_path / 'output'
    summary = quality_pipeline.run([source], output)
    manifest = json.loads((output / 'dump/dump-manifest.json').read_text())
    assert summary['preprocessing_failed_sources'] == 1
    assert manifest['dataset'] is None
    assert manifest['quality_starts'] == manifest['quality_nominal_starts'] == manifest['quality_recovery_starts'] == 0
    assert not (output / 'dump/rua_lerobot').exists()


def test_quality_rejection_is_not_processing_failure(tmp_path, monkeypatch):
    source = make_episode(tmp_path, 'unknown', task_steps=20)
    monkeypatch.setattr(quality_pipeline, 'replay', replay_fixture)
    original = quality_pipeline.classify

    def classify(source, oracle):
        labels = original(source, oracle)
        for label in labels:
            label['role'], label['verification'] = 'uncertain', 'unknown'
        return labels

    monkeypatch.setattr(quality_pipeline, 'classify', classify)
    summary = quality_pipeline.run([source], tmp_path / 'output')
    assert summary['preprocessing_failed_sources'] == 0
    assert summary['rejected_sources'] == 1
    assert summary['training_starts'] == 0


@pytest.mark.parametrize('failure_count,exit_code', [(0, 0), (1, 2)])
def test_cli_reports_partial_processing_failure(monkeypatch, failure_count, exit_code):
    monkeypatch.setattr('sys.argv', ['quality_pipeline', '/source', '--output-root', '/output'])
    monkeypatch.setattr(quality_pipeline, 'run', lambda *args: {'preprocessing_failed_sources': failure_count})
    assert quality_pipeline.main() == exit_code


def test_unexpected_runtime_error_is_not_silently_skipped(tmp_path, monkeypatch):
    source = make_episode(tmp_path, 'source', task_steps=20)
    def replay(*args):
        raise RuntimeError('Simulator dependency failed')
    monkeypatch.setattr(quality_pipeline, 'replay', replay)
    with pytest.raises(RuntimeError, match='Simulator dependency failed'):
        quality_pipeline.run([source], tmp_path / 'output')


def test_corrupt_trace_preserves_readable_events_but_rejects_episode(tmp_path):
    source = make_episode(tmp_path, 'corrupt_trace', task_steps=20)
    path = source / 'environment-steps.jsonl'
    lines = path.read_text().splitlines()
    path.write_text('\n'.join(lines[:3] + ['{broken'] + lines[3:]) + '\n')
    output = tmp_path / 'output'
    summary = quality_pipeline.run([source], output)
    assert summary['preprocessing_failed_sources'] == 1
    trace = [json.loads(line) for line in (output / 'dump/trace.jsonl').read_text().splitlines()]
    assert len([row for row in trace if row['kind'] == 'environment']) == len(lines)
    feedback = json.loads((output / 'dump/feedback.jsonl').read_text())
    assert feedback['rejection_code'] == 'source_invalid'
    assert 'line 4' in feedback['rejection_reason']
    assert feedback['wla_training_starts'] == 0


def test_missing_replay_file_is_reported_before_simulator_start(tmp_path):
    from data_dump.oracle_feedback import replay
    source = make_episode(tmp_path, 'missing_bddl', task_steps=20)
    with pytest.raises(FileNotFoundError, match='task.bddl'):
        replay(source, tmp_path / 'oracle')
