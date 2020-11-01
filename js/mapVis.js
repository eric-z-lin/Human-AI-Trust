/* * * * * * * * * * * * * *
*          MapVis          *
* * * * * * * * * * * * * */


class MapVis {

    constructor(parentElement, geoData, covidData, usaData) {
        this.parentElement = parentElement;
        this.geoData = geoData;
        this.covidData = covidData;
        this.usaData = usaData;
        this.displayData = [];

        // parse date method
        this.parseDate = d3.timeParse("%m/%d/%Y");

        this.initVis()
    }

    initVis() {
        let vis = this;
        vis.selectedCategory = $('#categorySelector').val();

        // Margins
        vis.margin = {top: 20, right: 20, bottom: 20, left: 20};
        vis.width = $("#" + vis.parentElement).width() - vis.margin.left - vis.margin.right;
        vis.height = $("#" + vis.parentElement).height() - vis.margin.top - vis.margin.bottom;

        // SVG drawing area
        vis.svg = d3.select("#" + vis.parentElement).append("svg")
            .attr("width", vis.width + vis.margin.left + vis.margin.right)
            .attr("height", vis.height + vis.margin.top + vis.margin.bottom)
            .attr("transform", "translate(" + vis.margin.left + "," + vis.margin.top + ")");

        // Zoom variable
        vis.viewpoint = {'width': 975, 'height': 610};
        vis.zoom = vis.width / vis.viewpoint.width;

        // Pass projection
        vis.path = d3.geoPath();

        // Add map to svg
        vis.map = vis.svg.append("g")
            .attr("class", "states")
            .attr("transform", `scale(${vis.zoom} ${vis.zoom})`);

        // Convert TopoJSON to GeoJSON
        vis.usa = topojson.feature(vis.geoData, vis.geoData.objects.states).features;

        // Draw states (transparent)
        vis.states = vis.map.selectAll(".state")
            .data(vis.usa)
            .enter().append("path")
            .attr('class', 'state')
            .attr("d", vis.path)
            .attr('fill-opacity', 0);

        // Color scale from yellow-green to blue (can change to red if want)
        vis.colorScale = d3.scaleQuantile()
            .range(d3.schemeYlGnBu[9]);

        // Legend
        vis.legend = vis.svg.append("g")
            .attr('class', 'legend')
            .attr('transform', `translate(${vis.margin.left}, ${vis.height - 60})`);

        // Tooltip
        vis.tooltip = d3.select("body").append('div')
            .attr('class', "tooltip")
            .attr('id', 'mapTooltip');

        vis.wrangleData(vis.selectedCategory);
    }

    wrangleData(selectedCategory){
        selectedCategory = $('#categorySelector').val();
        let vis = this;
        vis.selectedCategory = selectedCategory;

        // first, filter according to selectedTimeRange, init empty array
        let filteredData = [];

        // if there is a region selected
        if (selectedTimeRange.length !== 0){
            //console.log('region selected', vis.selectedTimeRange, vis.selectedTimeRange[0].getTime() )

            // iterate over all rows the csv (dataFill)
            vis.covidData.forEach( row => {
                // and push rows with proper dates into filteredData
                if (selectedTimeRange[0].getTime() <= vis.parseDate(row.submission_date).getTime() && vis.parseDate(row.submission_date).getTime() <= selectedTimeRange[1].getTime() ){
                    filteredData.push(row);
                }
            });
        } else {
            filteredData = vis.covidData;
        }

        // prepare covid data by grouping all rows by state
        let covidDataByState = Array.from(d3.group(filteredData, d =>d.state), ([key, value]) => ({key, value}))

        // have a look
        // console.log(covidDataByState)

        // init final data structure in which both data sets will be merged into
        vis.stateInfo = []

        // count for whatever selected category data
        let color_data_sum = 0;

        // merge
        covidDataByState.forEach( state => {

            // get full state name
            let stateName = nameConverter.getFullName(state.key)

            // init counters
            let newCasesSum = 0;
            let newDeathsSum = 0;
            let population = 0;

            // look up population for the state in the census data set
            vis.usaData.forEach( row => {
                if(row.state === stateName){
                    population += +row["2019"].replaceAll(',', '');
                }
            })

            // calculate new cases by summing up all the entries for each state
            state.value.forEach( entry => {
                newCasesSum += +entry['new_case'];
                newDeathsSum += +entry['new_death'];
            });

            // find data for color
            let color_data;
            if (selectedCategory === "absCases") {
                color_data = newCasesSum;
            }
            else if (selectedCategory === "absDeaths") {
                color_data = newDeathsSum;
            }
            else if (selectedCategory === "relCases") {
                color_data = (newCasesSum/population*100);
            }
            else if (selectedCategory === "relDeaths") {
                color_data = (newDeathsSum/population*100);
            }
            color_data_sum += color_data;

            // populate the final data structure
            vis.stateInfo.push(
                {
                    state: stateName,
                    population: population,
                    absCases: newCasesSum,
                    absDeaths: newDeathsSum,
                    relCases: (newCasesSum/population*100),
                    relDeaths: (newDeathsSum/population*100),
                    colorData: color_data
                }
            )
        })
        vis.dictionaryStateInfo =  {};
        vis.stateInfo.forEach( state => {
            vis.dictionaryStateInfo[state.state] = state;
        })

        vis.updateVis();
    }

    updateVis() {
        let vis = this;

        // Color Scale -- equally sized bins
        vis.colorScale
            .domain(Object.values(vis.stateInfo).map(d => d[vis.selectedCategory]));
        // Color Scale -- uncomment for unequally sized bins
        // vis.colorScale
        //     .domain(d3.extent(Object.values(vis.stateInfo).map(d => d[vis.selectedCategory])));

        // Update Legend -- using open source legend from Susie Lu
        let legend = d3.legendColor()
            .shapeHeight(15)
            .shapeWidth(45)
            .shapePadding(10)
            .labelOffset(10)
            .orient("horizontal")
            .labelWrap(30)
            .scale(vis.colorScale);

        // Format legend based on which category was selected
        if (vis.selectedCategory === "absCases") {
            legend.labelFormat('.2s')
        }
        else if (vis.selectedCategory === "absDeaths") {
            legend.labelFormat('.2s')
        }
        else if (vis.selectedCategory === "relCases") {
            legend.labelFormat('.2g')
        }
        else {
            legend.labelFormat('.2%')
        }


        vis.legend.call(legend);

        vis.states
            .attr('fill-opacity', 100)
            .attr('fill', function (d) {
                return vis.findColor(d);
            });


        // Tooltip mouseover and mouseout
        vis.states
            .on('mouseover', function(event, d){
                let stateInfo = vis.dictionaryStateInfo[d.properties.name]
                d3.select(this)
                    .attr('stroke-width', '2px')
                    .attr('stroke', 'black')
                    .attr('fill', 'rgba(255,160,160,0.62)')
                vis.tooltip
                    .style("opacity", 1)
                    .style("left", event.pageX + 20 + "px")
                    .style("top", event.pageY + "px")
                    .html(`
                         <div style="border: thin solid grey; border-radius: 5px; background: #e3c1ef; padding: 20px">
                             <h4>${stateInfo.state}<h4>
                             <h5> Population: ${d3.format('.2s')(stateInfo.population)}</h5>
                             <h5> Cases (absolute): ${d3.format('.2s')(stateInfo.absCases)}</h5>
                             <h5> Deaths (absolute): ${d3.format('.2s')(stateInfo.absDeaths)}</h5>
                             <h5> Cases (relative): ${d3.format('.2s')(stateInfo.relCases)}</h5>
                             <h5> Deaths (relative): ${d3.format('.2')(stateInfo.relDeaths)}</h5>
                         </div>`);
            })
            .on('mouseout', function(event, d){
                d3.select(this)
                    .attr('stroke-width', '0px')
                    .attr("fill", d => vis.findColor(d))
                vis.tooltip
                    .style("opacity", 0)
                    .style("left", 0)
                    .style("top", 0)
                    .html(``);
            });


    }

    findColor(d) {
        let vis = this;
        let colorData = vis.dictionaryStateInfo[d.properties.name][vis.selectedCategory];
        return vis.colorScale(colorData);
    }

}