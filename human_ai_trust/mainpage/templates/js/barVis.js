/* * * * * * * * * * * * * *
*      class BarVis        *
* * * * * * * * * * * * * */


class BarVis {

    constructor(parentElement, covidData, usaData, descending){
        this.parentElement = parentElement;
        this.covidData = covidData;
        this.usaData = usaData;
        this.descending = (descending === "top");
        this.displayData = [];
        this.selectedCategory = $('#categorySelector').val();

        // parse date method
        this.parseDate = d3.timeParse("%m/%d/%Y");

        this.initVis()
    }

    initVis(){
        let vis = this;
        vis.selectedCategory = $('#categorySelector').val();

        vis.margin = {top: 10, right: 10, bottom: 40, left: 45};
        vis.width = $("#" + vis.parentElement).width() - vis.margin.left - vis.margin.right;
        vis.height = $("#" + vis.parentElement).height() - vis.margin.top - vis.margin.bottom;

        // init drawing area
        vis.svg = d3.select("#" + vis.parentElement).append("svg")
            .attr("width", vis.width + vis.margin.left + vis.margin.right)
            .attr("height", vis.height + vis.margin.top + vis.margin.bottom)
            .append('g')
            .attr('transform', `translate (${vis.margin.left}, ${vis.margin.top})`);

        // add title
        let title_text = 'Top 10 Worst States';
        if (vis.descending == false) {
            title_text = 'Top 10 Best States'
        }
        vis.svg.append('g')
            .attr('class', 'title bar-title')
            .append('text')
            .text(title_text)
            .attr('transform', `translate(${vis.width / 2}, 10)`)
            .attr('text-anchor', 'middle');

        // Scales
        vis.xScale = d3.scaleBand()
            .range([0, vis.width - vis.margin.right])
            .paddingInner(0.1);
        vis.yScale = d3.scaleLinear()
            .range([vis.height, 0]);
        // Color scale from yellow-green to blue (can change to red if want)
        vis.colorScale = d3.scaleQuantile()
            .range(d3.schemeYlGnBu[9]);

        // Axes
        vis.xAxis = d3.axisBottom()
            .scale(vis.xScale)
            .tickSizeOuter(0);
        vis.yAxis = d3.axisLeft()
            .scale(vis.yScale)
            .tickSizeOuter(0);
        vis.svg.append("g")
            .attr("class", "x-axis axis")
            .attr("transform", "translate(0,"+vis.height+")");
        vis.svg.append("g")
            .attr("class", "y-axis axis");

        // append tooltip
        vis.tooltip = d3.select("body").append('div')
            .attr('class', "tooltip");

        this.wrangleData(vis.selectedCategory);
        vis.updateVis();
    }

    wrangleData(selectedCategory){
        let vis = this
        vis.selectedCategory = $('#categorySelector').val();

        // I think one could use a lot of the dataWrangling from dataTable.js here...

        // maybe a boolean in the constructor could come in handy ?
        /*
        if (vis.descending){
            vis.displayData.sort((a,b) => {return b[selectedCategory] - a[selectedCategory]})
        } else {
            vis.displayData.sort((a,b) => {return a[selectedCategory] - b[selectedCategory]})
        }

        console.log('final data structure', vis.displayData);

        vis.topTenData = vis.displayData.slice(0, 10)

        console.log('final data structure', vis.topTenData);
        */

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

            // populate the final data structure
            vis.stateInfo.push(
                {
                    state: stateName,
                    population: population,
                    absCases: newCasesSum,
                    absDeaths: newDeathsSum,
                    relCases: (newCasesSum/population*100),
                    relDeaths: (newDeathsSum/population*100)
                }
            )
        })

        if (vis.descending){
            vis.stateInfo.sort((a,b) => {return b[vis.selectedCategory] - a[vis.selectedCategory]})
        } else {
            vis.stateInfo.sort((a,b) => {return a[vis.selectedCategory] - b[vis.selectedCategory]})
        }
        vis.topTenData = vis.stateInfo.slice(0, 10)

        vis.updateVis()
    }

    updateVis(){
        let vis = this;
        vis.selectedCategory = $('#categorySelector').val();

        // Update color scale
        vis.colorScale
            .domain(Object.values(vis.stateInfo).map(function (d) {return d[vis.selectedCategory];}))

        // Update other scales
        vis.xScale
            .domain(vis.topTenData.map(d=>d.state))
        vis.yScale
            .domain([0,d3.max(vis.topTenData, d=>d[vis.selectedCategory])])

        // Enter update exit for barchart
        vis.bar = vis.svg.selectAll("rect")
            .data(vis.topTenData);
        vis.bar.exit().remove();
        vis.bar
            .on('mouseover', function(event, d){
                d3.select(this)
                    .attr('stroke-width', '2px')
                    .attr('fill', 'rgba(255,160,160,0.62)')
                vis.tooltip
                    .style("opacity", 1)
                    .style("left", event.pageX + 20 + "px")
                    .style("top", event.pageY + "px")
                    .html(`
                         <div style="background: #e3c1ef; padding: 20px">
                             <h4>${d.state}<h4>
                             <h5> Population: ${d3.format('.2s')(d.population)}</h5>
                             <h5> Cases (absolute): ${d3.format('.2s')(d.absCases)}</h5>
                             <h5> Deaths (absolute): ${d3.format('.2s')(d.absDeaths)}</h5>
                             <h5> Cases (relative): ${d3.format('.2s')(d.relCases)}</h5>
                             <h5> Deaths (relative): ${d3.format('.2')(d.relDeaths)}</h5>
                         </div>`);
            })
            .on('mouseout', function(event, d){
                d3.select(this)
                    .attr('stroke-width', '0px')
                    .attr("fill", d => vis.colorScale(d[vis.selectedCategory]))
                vis.tooltip
                    .style("opacity", 0)
                    .style("left", 0)
                    .style("top", 0)
                    .html(``);
            });
        vis.bar
            .enter().append("rect")
            .merge(vis.bar)
            .transition()
            .duration(800)
            .attr("height", d => vis.height - vis.yScale(d[vis.selectedCategory]))
            .attr("width", vis.xScale.bandwidth())
            .attr("stroke-width", "4")
            .attr("x", d => vis.xScale(d.state))
            .attr("y", d => vis.yScale(d[vis.selectedCategory]))
            .attr("fill", d => vis.colorScale(d[vis.selectedCategory]));

        // Update axes
        vis.svg.select(".x-axis")
            .call(vis.xAxis)
            .selectAll("text")
            .attr("text-anchor", "end")
            .attr("transform", "translate(5, 0), rotate(-15)");

        // Format yAxis based on which category was selected
        if (vis.selectedCategory === "absCases") {
            vis.yAxis.tickFormat(d3.format('.2s'))
        }
        else if (vis.selectedCategory === "absDeaths") {
            vis.yAxis.tickFormat(d3.format('.2s'))
        }
        else if (vis.selectedCategory === "relCases") {
            vis.yAxis.tickFormat(d3.format('.2g'))
        }
        else {
            vis.yAxis.tickFormat(d3.format('.2%'))
        }
        vis.svg.select(".y-axis")
            .call(vis.yAxis);


    }

}