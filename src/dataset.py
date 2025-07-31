import pandas as pd
import numpy as np

from pathlib import Path
from argparse import ArgumentParser


def fetch_training_data() -> pd.DataFrame:
    #return pd.read_csv('https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?network=TX_ASOS&station=RAS&data=tmpc&data=relh&data=drct&data=sped&year1=2025&month1=1&day1=1&year2=2025&month2=7&day2=24&tz=Etc%2FUTC&format=onlycomma&latlon=no&elev=no&missing=empty&trace=0.0001&direct=no&report_type=3')
    try:
        return pd.read_csv('training.csv').set_index('Unnamed: 0')
    except Exception:
        return pd.read_csv('https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?network=TX_ASOS&station=RAS&data=tmpc&data=relh&data=drct&data=sped&year1=2022&month1=1&day1=1&year2=2024&month2=12&day2=31&tz=Etc%2FUTC&format=onlycomma&latlon=no&elev=no&missing=empty&trace=0.0001&direct=no&report_type=3')

def fetch_testing_data() -> pd.DataFrame:
    #return pd.read_csv('https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?network=TX_ASOS&station=RAS&data=tmpc&data=relh&data=drct&data=sped&year1=2025&month1=1&day1=1&year2=2025&month2=7&day2=24&tz=Etc%2FUTC&format=onlycomma&latlon=no&elev=no&missing=empty&trace=0.0001&direct=no&report_type=3')
    try:
        return pd.read_csv('testing.csv').set_index('Unnamed: 0')
    except Exception:
        return pd.read_csv('http://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?network=TX_ASOS&station=RAS&data=tmpc&data=relh&data=drct&data=sped&year1=2025&month1=1&day1=1&year2=2025&month2=7&day2=24&tz=Etc%2FUTC&format=onlycomma&latlon=no&elev=no&missing=empty&trace=0.0001&direct=no&report_type=3')


def mk_rolling_timeseries(df: pd.DataFrame,
                          data_cols: list[str],
                          steps: list[int]) -> pd.DataFrame:
    shifted = [
        df[data_cols].shift(-step).rename(columns={col: f'{col}-{step}' for col in data_cols})
        for step in steps]

    return pd.concat([df, *shifted], axis=1)


def mk_txy(data: pd.DataFrame) -> tuple[pd.Series, np.array, np.array]:
    data = data.assign(
        u=lambda df: np.cos(np.deg2rad(df['drct']))*df['sped'],
        v=lambda df: np.sin(np.deg2rad(df['drct']))*df['sped'],
    ).drop(columns=['drct', 'sped', 'station'])
    timeseries = mk_rolling_timeseries(data, ['tmpc','relh','u','v'], [12,13,14,15,18,21,24])
    timeseries = timeseries.dropna()

    xs = timeseries.values[:,5:].astype(float)
    ys = timeseries.values[:,1].reshape(-1,1).astype(float)
    return timeseries['valid'], xs, ys


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('output_dir', type=str)

    args = parser.parse_args()


    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    fetch_training_data().to_csv(output_dir / 'training.csv')
    fetch_testing_data().to_csv(output_dir / 'testing.csv')