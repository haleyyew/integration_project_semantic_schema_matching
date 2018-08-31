const program = require('commander');
const rp = require('request-promise');

const FORMATS = ["json", "csv"];

const RegisteredSources = [
	{name: "Surrey", fetchResourceHandler: (st) => {return surreyGetData(st)}}
]


const requestPromise = (req) => {
  return rp({
    uri: req,
    json: true,
    resolveWithFullResponse: true
  }).then(function(response) {
      return response;
  });
}

const surreyGetData = function() {
  requestPromise("http://data.surrey.ca/api/3/action/package_search?q=")
  .then( (response) => {
    	console.log("response.count: " + response.body.result.count);
			// resources = response.body.result.results.map( (resource) => {
			// 	formats = resource.resources.map( (resource) => {
			// 		return {format: resource.format.toLowerCase(), url: resource.url }
			// 	} )
			// 	return {tag: resource.name.toLowerCase(), formats, src: "surrey"}
			// } )

		console.log("response.results: " + response.body.result.results);	
		return response;
  });
}

program
  .version('1');

program
//  .command('match <tag>')
  .command('download')
  .description('Project: Schema Integration for Heterogenous Data Sources')
  .action(function(){
    console.log('calling surreyGetData()');
    surreyGetData();
  })

program.parse(process.argv);



