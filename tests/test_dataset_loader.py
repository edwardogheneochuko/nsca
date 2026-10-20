import json

import pytest

from dataset_loader import DatasetError, load_dataset, normalise_symptom, vectorize


def write_csv(path, rows, header="Disease,Symptom_1,Symptom_2,Symptom_3"):
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    return path


def valid_rows():
    a = [f"DiseaseA,sym one,sym_two,{'' if i % 2 else 'sym-three'}" for i in range(6)]
    b = [f"DiseaseB,sym_two,sym four,{'' if i % 2 else 'sym_five'}" for i in range(6)]
    return a + b


def test_missing_file(tmp_path):
    with pytest.raises(DatasetError, match="not found"):
        load_dataset(tmp_path / "nope.csv", tmp_path / "out")


def test_missing_disease_column(tmp_path):
    p = write_csv(tmp_path / "d.csv", ["x,y"], header="Name,Symptom_1")
    with pytest.raises(DatasetError, match="Disease"):
        load_dataset(p, tmp_path / "out")


def test_no_symptom_columns(tmp_path):
    p = write_csv(tmp_path / "d.csv", ["x,y"], header="Disease,Other")
    with pytest.raises(DatasetError, match="symptom columns"):
        load_dataset(p, tmp_path / "out")


def test_normalisation():
    assert normalise_symptom(" Dischromic _Patches ") == "dischromic_patches"
    assert normalise_symptom("high-fever") == "high_fever"


def test_cleaning_dedup_and_processed_outputs(tmp_path):
    uniq = [f"DiseaseA,a{i},a{i+1}," for i in range(6)] + [f"DiseaseB,b{i},b{i+1}," for i in range(6)]
    rows = uniq + [uniq[0], uniq[7], ",x,y,", "DiseaseB,,,"]
    ds = load_dataset(write_csv(tmp_path / "d.csv", rows), tmp_path / "processed")
    assert ds.n_raw_rows == 16 and ds.n_rows_dropped == 2
    assert ds.n_duplicates_removed == 2 and ds.n_rows == 12
    assert ds.processed_written
    manifest = json.loads((tmp_path / "processed" / "manifest.json").read_text())
    assert manifest["original_sha256"] == ds.sha256 and manifest["n_training_rows"] == ds.n_rows
    for f in ("cases.csv", "features.npz", "feature_names.json"):
        assert (tmp_path / "processed" / f).exists()


def test_too_few_rows(tmp_path):
    with pytest.raises(DatasetError, match="usable unique cases"):
        load_dataset(write_csv(tmp_path / "d.csv", valid_rows()), tmp_path / "out")


def test_conflicting_patterns_counted(tmp_path):
    rows = [f"DiseaseA,s{i},s{i+1}," for i in range(6)] + [f"DiseaseB,s{i},s{i+1}," for i in range(6)]
    ds = load_dataset(write_csv(tmp_path / "d.csv", rows), tmp_path / "out")
    assert ds.n_conflicting_patterns == 6 and ds.n_rows == 12


def test_original_not_modified_and_vectorize_consistent(tmp_path):
    p = write_csv(tmp_path / "d.csv", [f"DiseaseA,s{i},s{i+1}," for i in range(6)] +
                  [f"DiseaseB,t{i},t{i+1}," for i in range(6)])
    before = p.read_bytes()
    ds = load_dataset(p, tmp_path / "out")
    assert p.read_bytes() == before
    v1 = vectorize(["S1", "t2", "unknown"], ds.symptom_index)
    v2 = vectorize(["t2", "s1"], ds.symptom_index)
    assert (v1 == v2).all() and v1.sum() == 2 and len(v1) == ds.n_symptoms


def test_unwritable_processed_dir_does_not_fail(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    p = write_csv(tmp_path / "d.csv", [f"DiseaseA,s{i},s{i+1}," for i in range(6)] +
                  [f"DiseaseB,t{i},t{i+1}," for i in range(6)])
    ds = load_dataset(p, blocker / "sub")
    assert ds.processed_written is False
