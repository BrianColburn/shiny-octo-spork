import dataset
import keras
import pandas as pd
import model
import plotly.express as px
import plotly.subplots
import plotly.graph_objs as go
import ipywidgets.widgets as widgets
import ipywidgets.embed
from pathlib import Path

from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('model_name', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('time_series_name', type=str)
    parser.add_argument('scatter_name', type=str)
    parser.add_argument('report_name', type=str)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(exist_ok=True)

    ts, xs, ys = dataset.mk_txy(dataset.fetch_testing_data())

    nn = keras.models.load_model(args.model_name)
    print(nn.summary())

    outputs = pd.DataFrame(nn.predict(xs), columns=['mu', 'sigma'], index=ts).assign(
        upper_bound=lambda df: df['mu'] + df['sigma'],
        lower_bound=lambda df: df['mu'] - df['sigma'],
    )

    fig = plotly.subplots.make_subplots(
        rows=2,
        shared_xaxes=True,
        row_heights=[1,5],
    )
    fig.add_trace(
        go.Scatter(
            name='sigma',
            x=outputs.index,
            y=outputs['sigma'],
            line=dict(color='rgb(0,100,80)'),
            mode='lines'
        ),
        col=1,
        row=1,
    ).add_traces([
            go.Scatter(
                name='mu',
                x=outputs.index,
                y=outputs['mu'],
                line=dict(color='rgb(0,100,80)'),
                mode='lines'
            ),
            go.Scatter(
                name='Upper Bound',
                x=outputs.index,
                y=outputs['upper_bound'],
                mode='lines',
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False
            ),
            go.Scatter(
                name='Lower Bound',
                x=outputs.index,
                y=outputs['lower_bound'],
                marker=dict(color="#444"),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(68, 68, 68, 0.3)',
                fill='tonexty',
                showlegend=False
            ),
            go.Scatter(
                name='Target',
                x=outputs.index[:-12],
                y=ys[12:,0],
                marker=dict(color="red"),
                mode='lines',
            ),
        ],
        cols=[1,1,1,1],
        rows=[2,2,2,2],
    ).write_html(args.time_series_name)

    print(f'Wrote {args.time_series_name}')

    df = pd.DataFrame({'y_true': ys[:,0], 'y_pred': nn(xs)[:,0]})
    scatter_fig = px.scatter(df, x='y_true', y='y_pred', width=600, height=500, title='Predictions vs Observations')

    scatter_fig.write_html(args.scatter_name)
    print(f'Wrote {args.scatter_name}')

    keras.utils.plot_model(nn, show_shapes=True, expand_nested=True, show_layer_activations=True, rankdir='TB', show_layer_names=True)


    # The HTML report workflow here is more painful than necessary.
    # Doing this in a Jupyter notebook would allow for a much nicer dev workflow, but would mean we lose the ability to easily generate the report in a script.
    # It is possible to automate Jupyter notebooks (and even publication-ready papers, like Quarto), but that is outside the scope of this demo.

    html_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
</head>
<body>
{snippet}
<plot_1>
<plot_2>
</body>
</html>
    '''

    ipywidgets.embed.embed_minimal_html(args.report_name, template=html_template, views=[widgets.VBox([
        widgets.HTML('''<p style="font: 14pt sans-serif">Here's an example of an html file with multiple embedded widgets. Ideal for creating and sharing reports, since the recipient only needs a modern web browser.</p>'''),
        widgets.Image(value=Path('model.png').read_bytes(), width=1000),
    ])])
    Path('model.png').unlink()
    report_path = Path(args.report_name)
    report_path.write_text(report_path.read_text().replace(
            '<plot_1>',
            fig.to_html(full_html=False)
        ).replace(
            '<plot_2>',
            scatter_fig.to_html(full_html=False)
        ), encoding='utf8')
    print(f'Wrote {args.report_name}')

