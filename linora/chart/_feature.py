import pyecharts as pe

__all__ = ['feature_corr_map']

def feature_corr_map(df, jupyter=True, path="Feature_Correlation_Map.html"):
    t = [[nor, row, j] for nor, i in enumerate(df.corr().round(3).values.tolist()) for row, j in enumerate(i)]
    heatmap = (
            pe.charts.HeatMap()
            .add_xaxis(
                xaxis_data=df.columns.tolist())
            .add_yaxis(
                "Feature Correlation", 
                yaxis_data=df.columns.tolist(), 
                value=t,
                label_opts=pe.options.LabelOpts(is_show=True, color="#fff", position='inside', horizontal_align="50%")
            )
            .set_global_opts(
                title_opts=pe.options.TitleOpts(title="Feature Correlation Map"),
                xaxis_opts=pe.options.AxisOpts(
                    type_="category", 
                    name='Actual', 
                    splitarea_opts=pe.options.SplitAreaOpts(
                        is_show=True, areastyle_opts=pe.options.AreaStyleOpts(opacity=1))),
                yaxis_opts=pe.options.AxisOpts(
                    type_="category", 
                    name='Predict', 
                    splitarea_opts=pe.options.SplitAreaOpts(
                        is_show=True, areastyle_opts=pe.options.AreaStyleOpts(opacity=1))),
                visualmap_opts=pe.options.VisualMapOpts(min_=0, max_=int(a.max().max()))))
    return  heatmap.render_notebook() if jupyter else heatmap.render(path)
