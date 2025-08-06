import plotly.express as px
import re
import plotly.graph_objects as go
import plotly.subplots

import pandas as pd
import numpy as np


def convert_col(s: pd.Series) -> pd.Series:
    try:
        return pd.to_numeric(s)
    except Exception:
        return s


def facet_fix_xaxes(plot, log_axes=set(), variable_name=None):
    if variable_name is None:
        scatter_titles = {t.yaxis: re.sub(r'.*params_(?P<param>.+?)(<br>.+|$)', r'\g<param>', t.hovertemplate)
                        for t in plot.data if type(t).__name__ in ['Scatter', 'Scattergl']}
        hist_titles = {t.yaxis: re.sub(r'.*params_(?P<param>.+?)(<br>.+|$)', r'\g<param>', t.hovertemplate)
                    for t in plot.data if type(t).__name__ == 'Histogram'}
        #plot.update_xaxes(matches=None)
        for anchor, title in scatter_titles.items():
            if title in log_axes:
                plot.update_xaxes(selector=dict(anchor=anchor), title_text=title, type='log', matches=None)
            else:
                plot.update_xaxes(selector=dict(anchor=anchor), title_text=title, matches=None)
            plot.update_annotations(selector=dict(text=f'param=params_{title}'), text='')
        for anchor, title in hist_titles.items():
            #plot.update_xaxes(selector=dict(anchor=anchor), title_text='', type=('log' if title in log_axes else 'linear'))
            if title in log_axes:
                plot.update_xaxes(selector=dict(anchor=anchor), title_text='', matches=None)
            else:
                plot.update_xaxes(selector=dict(anchor=anchor), title_text='', matches=None)
            plot.update_yaxes(selector=dict(anchor=anchor.replace('y','x')), matches=None)
            for dat in plot.data:
                if dat.yaxis == anchor and title in log_axes:
                    old_x = dat.x
                    try:
                        dat.x = np.log10(np.array(dat.x).astype(np.float64))
                    except Exception as e:
                        print('current dat.x:',dat.x)
                        print('old dat.x:', old_x)
                        print(type(old_x))
                        print(old_x.dtype)
                        raise e
        return plot
    else:
        scatter_titles = {t.yaxis: re.sub(f'.*{variable_name}(?P<param>.+?)(<br>.+|$)', r'\g<param>', t.hovertemplate)
                        for t in plot.data if type(t).__name__ in ['Scatter', 'Scattergl']}
        hist_titles = {t.yaxis: re.sub(f'.*{variable_name}(?P<param>.+?)(<br>.+|$)', r'\g<param>', t.hovertemplate)
                    for t in plot.data if type(t).__name__ == 'Histogram'}
        #plot.update_xaxes(matches=None)
        for anchor, title in scatter_titles.items():
            if title in log_axes:
                plot.update_xaxes(selector=dict(anchor=anchor), title_text=title, type='log', matches=None)
            else:
                plot.update_xaxes(selector=dict(anchor=anchor), title_text=title, matches=None)
            plot.update_annotations(selector=dict(text=f'{variable_name}{title}'), text='')
        for anchor, title in hist_titles.items():
            #plot.update_xaxes(selector=dict(anchor=anchor), title_text='', type=('log' if title in log_axes else 'linear'))
            if title in log_axes:
                plot.update_xaxes(selector=dict(anchor=anchor), title_text='', matches=None)
            else:
                plot.update_xaxes(selector=dict(anchor=anchor), title_text='', matches=None)
            plot.update_yaxes(selector=dict(anchor=anchor.replace('y','x')), matches=None)
            for dat in plot.data:
                if dat.yaxis == anchor and title in log_axes:
                    dat.x = np.log10(dat.x)
        return plot


def fmt_presentation_figure(fig):
    """Format presentation figure according to CBI standards."""
    return fig.update_layout(margin=dict(t=50, r=0, b=0, l=0), title_font_size=32).update_yaxes(title_font_size=24, tickfont_size=16).update_xaxes(title_font_size=24, tickfont_size=16)#.to_image(format='png', scale=2)


def mk_slice_plot(study,
                  invert=False,
                  filter=None,
                  title='Slice Plot',
                  color='number',
                  target='value',
                  target_name=None,
                  log_cols=['params_l2_regularization', 'params_learning_rate'],
                  dist_plot='histogram',
                  title_append_n=True,
                  debug=False
                 ):
    if type(study) == pd.DataFrame:
        if 'state' in study.columns:
            df = study.query('state == "COMPLETE"')
        else:
            df = study
    else:
        df = study.trials_dataframe().query('state == "COMPLETE"')
    
    if isinstance(target, str):
        if target not in df.columns:
            raise ValueError(f'Target column "{target}" missing from dataframe!\nColumns: {df.columns}')
        elif target_name is None:
            target_name = target
    #elif isinstance(target, pd.Series) and target_name not in df.columns:
    #    df[target_name] = target
    #    target = target_name
    else:
        raise ValueError(f'Invalid target "{target}" (type: {type(target)})!')
    
    if debug:
        print('DataFrame:\n',df.info())

    bad_df = None

    if filter is not None:
        filtered_df = filter(df)
        good_indices = filtered_df.index
        bad_indices = df.index.difference(good_indices)
        bad_df = df.loc[bad_indices]
        df = filtered_df
        if debug:
            print('Filtered:\n',df.info())

    trial_count = len(df)
    if bad_df is not None:
        trial_count += len(bad_df)

    nonna_trial_count = len(df[target].replace([np.inf,-np.inf],np.nan).dropna())

    if invert:
        df[target] = -df[target]
    
    #cols = [col for col in df.columns if ('params' in col and 'hidden' not in col) or 'kfold' in col or col in [color, target]]
    cols = [col for col in df.columns if ('params' in col) or 'kfold' in col or col in [color, target]]
    log_cols = [col for col in cols if col in log_cols]
    nonna = df[cols].dropna(axis=1)
    melted = nonna.melt(id_vars=nonna.columns.intersection([target, color, 'user_attrs_kfold']), var_name='param', value_name='param_value')

    if debug:
        print('Melted:\n',melted.info())
    
    if callable(title):
        title = title(df=df, trial_count=trial_count, nonna_trial_count=nonna_trial_count)
    
    if title_append_n:
        title += f''' (n={nonna_trial_count}/{trial_count})'''
    
    if color not in nonna.columns:
        color = None
    
    #num_params = len([c for c in cols + log_cols if 'param' in c])

    #fig = plotly.subplots.make_subplots(
    #    rows=2,
    #    cols=num_params,
    #)

    #for col_ix, col_name in enumerate(cols + log_cols):
    #    col_dtype = df[col_name].dtype
    #    plot_func = None

    #    match col_dtype:
    #        case pd.Int64Dtype():
    #            plot_func = px.violin
    #        case pd.Float64Dtype() | pd.StringDtype():
    #            plot_func = px.scatter

    #    plot = px.scatter(df, y=target, x=df[col_name], marginal_x=dist_plot, height=300, width=1000)
    #    fig.add_trace(plot.data[1], row=1, col=col_ix+1)
    #    plot = plot_func(df, y=target, x=df[col_name], height=300, width=1000)
    #    fig.add_trace(plot.data[0], row=2, col=col_ix+1)
    
    fig = fmt_presentation_figure(facet_fix_xaxes(
        plot=px.scatter(melted, y=target, x='param_value',
                        facet_col='param',
                        color=color,
                        color_continuous_scale=['white', 'blue'],
                        title=title,
                        labels={target: target_name, 'number': 'Trial Num', 'user_attrs_kfold': 'K-Fold Rotation'},
                        marginal_x=dist_plot,
                        height=300,
                        width=1000
                       ),
        log_axes=[col.replace('params_','') for col in log_cols]
    ).update_traces(
        marker_line=dict(width=1, color='gray'),
        selector=dict(mode='markers')).update_traces(
        selector=dict(type='histogram'),
        marker_color='blue',
        nbinsx=10,
        bingroup=None)).update_xaxes(
        title_font_size=16)
    
    if debug:
        print('Created Figure')
    
    na_mask = df[target].replace([np.inf,-np.inf],np.nan).isna()
    ix = 0
    y_range = df[target].min(), df[target].max()

    lines = []
    for col in nonna.columns:
        if col in {target, color, 'number', 'user_attrs_kfold'}:
            continue
        if debug:
            print(f'Adding bad HP lines for {col}')
        ix += 1
        #print(f'Adding lines to {col}({ix})')
        
        for x in df[col][na_mask].values:
            if np.isnan(x):
                continue
            if debug:
                print(f'Adding nan line to {col}({ix}) at {x}')
            fig.add_vline(x=x, col=ix, row=1, opacity=0.2, line_color='red')
        
        if bad_df is not None:
            for x in bad_df[col].values:
                if isinstance(x, float) and np.isnan(x):
                    continue
                if debug:
                    print(f'Adding filtered line to {col} at {x}')
                lines.append(go.layout.Shape(
                    type='line',
                    x0=x,
                    x1=x,
                    xref=f'x{ix}' if ix > 0 else 'x',
                    y0=y_range[0]*0.98,
                    y1=y_range[1]*1.02,
                    opacity=0.2,
                    line_color='red',
                    line_width=1,
                    line_dash='dash',
                    layer='below',
                ))
                #fig.add_vline(x=x, col=ix, row=1, opacity=0.2, line_color='red', line_dash='dash')
    fig.update_layout(shapes=lines)

    return fig
