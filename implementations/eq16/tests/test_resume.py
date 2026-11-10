import csv
import tempfile
from pathlib import Path
from run_benchmark import read_partial_csv

def test_resume_csv_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'validation.csv'
        with p.open('w', newline='') as f:
            wr=csv.DictWriter(f,fieldnames=['shift','epsilon','seed','reward','steps','time_sec'])
            wr.writeheader(); wr.writerow(dict(shift='low_mass',epsilon='0.0001',seed='700126',reward='23.5',steps='150',time_sec='551.0'))
        rows=read_partial_csv(p,{'shift','epsilon','seed','reward','steps','time_sec'})
        assert rows[0]['seed']==700126
        assert rows[0]['epsilon']==0.0001
        assert rows[0]['reward']==23.5
