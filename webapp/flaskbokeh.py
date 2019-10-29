from flask import Flask, render_template

from bokeh.embed import server_document
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Slider, Button
from bokeh.plotting import figure
from bokeh.server.server import Server
from bokeh.themes import Theme
from tornado.ioloop import IOLoop

from bokeh.sampledata.sea_surface_temperature import sea_surface_temperature

app = Flask(__name__)


def modify_doc(doc):
    tools = "pan,wheel_zoom,lasso_select,reset"
    fig = figure(title='Select points',
                 plot_width=300, plot_height=200,tools=tools)

    import numpy as np
    x = np.linspace(0,10,100)
    y = np.random.random(100) + x

    import pandas as pd
    data = pd.DataFrame(dict(x=x, y=y))

    # define data source
    src = ColumnDataSource(data)
    # define plot
    fig.circle(x='x', y='y', source=src)

    # define interaction
    def print_datapoints():
        indices = src.selected.indices
        results = data.iloc[indices]
        print(results)
        resultsDict=results.to_dict()['x']
        resultString=str(resultsDict)


    btn = Button(label='Selected points', button_type='success')
    btn.on_click(print_datapoints)

    doc.add_root(column(btn, fig))

    doc.theme = Theme(filename="theme.yaml")


@app.route('/', methods=['GET'])
def bkapp_page():
    script = server_document('http://localhost:5006/bkapp')
    return render_template("embed.html", script=script, template="Flask")


def bk_worker():
    # Can't pass num_procs > 1 in this configuration. If you need to run multiple
    # processes, see e.g. flask_gunicorn_embed.py
    server = Server({'/bkapp': modify_doc}, io_loop=IOLoop(), allow_websocket_origin=["localhost:8000"])
    server.start()
    server.io_loop.start()

from threading import Thread
Thread(target=bk_worker).start()

if __name__ == '__main__':
    print('Opening single process Flask app with embedded Bokeh application on http://localhost:8000/')
    print()
    print('Multiple connections may block the Bokeh app in this configuration!')
    print('See "flask_gunicorn_embed.py" for one way to run multi-process')
    app.run(port=8000, debug=True)