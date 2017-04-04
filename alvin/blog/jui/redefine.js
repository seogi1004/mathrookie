jui.redefine("chart.brush.polygon.scatter3d",
	[ "util.base", "util.math", "util.color", "chart.polygon.point" ],
	function(_, MathUtil, ColorUtil, PointPolygon) {

	/**
	 * @class chart.brush.polygon.scatter3d
	 * @extends chart.brush.polygon.core
	 */
	var PolygonScatter3DBrush = function() {
		this.createScatter = function(data, dataIndex) {
			var color = this.color(dataIndex),
				r = this.brush.size / 2,
				x = this.axis.x(data.x),
				y = this.axis.y(data.y),
				z = this.axis.z(data.z);

			return this.createPolygon(new PointPolygon(x, y, z), function(p) {
				var elem = this.chart.svg.circle({
					r: (r * MathUtil.scaleValue(z, 0, this.axis.depth, 1, p.perspective)) || r,
					fill: data.color || color,
					"fill-opacity": this.chart.theme("polygonScatterBackgroundOpacity"),
					cx: p.vectors[0].x,
					cy: p.vectors[0].y
				});

				return elem;
			});
		}

		this.draw = function() {
			var g = this.chart.svg.group(),
				datas = this.listData();

			for(var i = 0; i < datas.length; i++) {
				g.append(this.createScatter(datas[i], i));
			}

			return g;
		}
	}

	PolygonScatter3DBrush.setup = function() {
		return {
			/** @cfg {Number} [size=7]  Determines the size of a starter. */
			size: 4,
			/** @cfg {Boolean} [clip=false] If the brush is drawn outside of the chart, cut the area. */
			clip: false
		};
	}

	return PolygonScatter3DBrush;
}, "chart.brush.polygon.core");

jui.redefine("chart.brush.polygon.face3d",
	[ "util.base", "util.math", "util.color", "chart.polygon.point" ],
	function(_, MathUtil, ColorUtil, PointPolygon) {

	/**
	 * @class chart.brush.polygon.scatter3d
	 * @extends chart.brush.polygon.core
	 */
	var PolygonFace3DBrush = function() {
		this.createScatter = function(g, elem, data, color) {
			var r = this.brush.size / 2,
				x = this.axis.x(data.x),
				y = this.axis.y(data.y),
				z = this.axis.z(data.z);

			return this.createPolygon(new PointPolygon(x, y, z), function(p) {
              	g.append(this.chart.svg.circle({
					r: (r * MathUtil.scaleValue(z, 0, this.axis.depth, 1, p.perspective)) || r,
					fill: color,
					"fill-opacity": this.chart.theme("polygonScatterBackgroundOpacity"),
					cx: p.vectors[0].x,
					cy: p.vectors[0].y
				}));
              	
              	elem.point(p.vectors[0].x, p.vectors[0].y);
			});
		}

		this.draw = function() {
			var g = this.chart.svg.group(),
				datas = this.listData(),
                color = this.color(0);
          
            var elem = this.chart.svg.polygon({
                fill: color,
                "fill-opacity": 0.5
            });
      
          	g.append(elem);

			for(var i = 0; i < datas.length; i++) {
				this.createScatter(g, elem, datas[i], color);
			}

			return g;
		}
	}

	PolygonFace3DBrush.setup = function() {
		return {
			/** @cfg {Number} [size=7]  Determines the size of a starter. */
			size: 4,
			/** @cfg {Boolean} [clip=false] If the brush is drawn outside of the chart, cut the area. */
			clip: false
		};
	}

	return PolygonFace3DBrush;
}, "chart.brush.polygon.core");

jui.redefine("chart.brush.polygon.line3d",
	[ "util.base", "util.color", "util.math", "chart.polygon.line" ],
	function(_, ColorUtil, MathUtil, LinePolygon) {

	/**
	 * @class chart.brush.polygon.line3d
	 * @extends chart.brush.polygon.core
	 */
	var PolygonLine3DBrush = function() {
		this.createLine = function(data, dataIndex) {
			var color = this.color(dataIndex),
				x1 = this.axis.x(data.sx),
				y1 = this.axis.y(data.sy),
				z1 = this.axis.z(data.sz),
				x2 = this.axis.x(data.x),
				y2 = this.axis.y(data.y),
				z2 = this.axis.z(data.z);

			var elem = this.chart.svg.polygon({
				fill: data.color || color,
				"fill-opacity": this.chart.theme("polygonLineBackgroundOpacity"),
				stroke: data.color || color,
				"stroke-opacity": this.chart.theme("polygonLineBorderOpacity")
			});

			var points = [
				new LinePolygon(x1, y1, z1),
				new LinePolygon(x2, y2, z2)
			];

			for(var i = 0; i < points.length; i++) {
				this.createPolygon(points[i], function(p) {
					var vector = p.vectors[0];
					elem.point(vector.x, vector.y);
				});
			}

			return elem;
		}

		this.draw = function() {
			var g = this.chart.svg.group(),
				datas = this.listData();

			for(var i = 0; i < datas.length; i++) {
				g.append(this.createLine(datas[i], i));
			}

			return g;
		}
	}

	PolygonLine3DBrush.setup = function() {
		return {
			/** @cfg {Boolean} [clip=false] If the brush is drawn outside of the chart, cut the area. */
			clip: false
		};
	}

	return PolygonLine3DBrush;
}, "chart.brush.polygon.core");