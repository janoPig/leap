import numpy as np, pandas as pd
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from HROCH import SymbolicRegressor
import sys, time, argparse, json

TEST_SIZE = 0.5

def evaluate_target(target, random_seed, out_dir):

    df_train = pd.read_parquet('./data/df_sample.parquet')
    df_sub_tmp = pd.read_csv('./data/sample_submission.csv.zip', nrows=1, compression='zip')
    targets = [x for x in df_sub_tmp.columns.tolist() if x != 'sample_id']
    if not target in targets:
        print(f'Invalid target {target}')
        sys.exit(1)

    features = [x for x in df_train.columns.tolist() if x not in ['sample_id']+targets]
    X = df_train.drop(targets + ['sample_id'], axis = 1)
    y = df_train[target]
    y=(y-np.min(y)+1e-10)/(np.max(y)-np.min(y)+1e-10)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=random_seed)

    lr_est = LinearRegression()
    lr_est.fit(X_train, y_train)
    lr_score_train, lr_score_test = lr_est.score(X_train, y_train), lr_est.score(X_test, y_test)
    print(f'LR score: train {lr_score_train} test {lr_score_test}')
    
    dt_est = DecisionTreeRegressor()
    dt_est.fit(X_train, y_train)
    dt_score_train, dt_score_test = dt_est.score(X_train, y_train), dt_est.score(X_test, y_test)
    print(f'DT score: train {dt_score_train} test {dt_score_test}')

    cb_est = CatBoostRegressor(silent=True)
    cb_est.fit(X_train, y_train)
    cb_score_train, cb_score_test = cb_est.score(X_train, y_train), cb_est.score(X_test, y_test)
    print(f'CB score: train {cb_score_train} test {cb_score_test}')

    MATH = {'nop': 0.1, 'add': 0.1, 'sub': 0.03, 'mul': 1.0,
            'div': 0.1, 'sq2': 0.15, 'pow': 0.001, 'exp': 0.3,
            'log': 0.01, 'sqrt': 0.1,'tanh': 0.001}
    algo_settings = {'neighbours_count':15, 'alpha':0.15, 'beta':0.65, 'pretest_size':16, 'sample_size':1024}
    population_settings = {'size': 64, 'tournament':4}
    sr_est = SymbolicRegressor(
        random_state=random_seed,
        num_threads=4,
        precision='f64',
        time_limit=5.0,
        algo_settings=algo_settings,
        population_settings=population_settings,
        feature_probs = cb_est.feature_importances_,
        warm_start=True,
        problem=MATH,
        )
    for n in range(60):
        sr_est.fit(X_train, y_train)
        sr_score_train, sr_score_test = sr_est.score(X_train, y_train), sr_est.score(X_test, y_test)
        models = sr_est.get_models()[:10]
        for m in models:
            m_score = m.score(X_test, y_test)
            print(m_score)

        print(f'SR score: train {sr_score_train} test {sr_score_test}')
        print(sr_est.sexpr_)
        #sr_models[target] = (sr_score_test, sr_est)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Fit estimators on sample dataset')
    parser.add_argument('--target', type=str, help='Target')
    parser.add_argument('--random_seed', type=int, help='Random seed index')
    parser.add_argument('--out_dir', type=str, help='Output dir')
    args = parser.parse_args()
    return args.target, args.random_seed, args.out_dir

if __name__ == '__main__':
    target, random_seed, out_dir = parse_arguments()

    evaluate_target(target, random_seed, out_dir )
    
    
