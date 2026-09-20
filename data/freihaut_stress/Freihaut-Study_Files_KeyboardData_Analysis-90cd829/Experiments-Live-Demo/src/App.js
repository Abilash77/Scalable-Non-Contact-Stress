// React imports
import React, {Component} from 'react';

// import the css file
import './App.css';

//CSS Framework Import
import 'bulma/css/bulma.css';
import 'bulma-extensions/dist/css/bulma-extensions.min.css';

//import components
import Navbar from './General-Components/Navbar';
import DemoStartPage from "./General-Components/DemoStartPage";
import OnlineExpNavigator from "./Online-Study/Online-Exp-Navigator";
import LabExpNavigator from './Lab-Study/Lab-Exp-Navigator';
import EdgeWarning from "./General-Components/EdgeWarning";
import InternetExplorerWarning from "./General-Components/InternetExplorerWarning";

export default class App extends Component {

  constructor(props) {
    super(props);

    this.state = {
      exp: false,
      experimentalPhase: 0,
      numberOfExperimentPages: 0,
      skipButtonDisabled: true,
      isInternetExplorer: false,
      isOldEdge: false,
      badBrowser: false,
      onlineStudy: {restarts: 0}
    };

    this.startExperiment = this.startExperiment.bind(this);
    this.nextPage = this.nextPage.bind(this);
    this.getExpLength = this.getExpLength.bind(this);
    this.toggleSkipButton = this.toggleSkipButton.bind(this);
    this.startExpWithEdge = this.startExpWithEdge.bind(this);

  }

  componentDidMount() {

    // check if the participant uses windows edge or internet explorer to view the study
    // Internet Explorer 6-11
    const isIE = /*@cc_on!@*/false || !!document.documentMode;
    // Edge 20+ (newest edge browser works fine!)
    const isOldEdge = !isIE && !!window.StyleMedia;

  // set state based on the used browser (IE and Edge are not allowed because they do not reliably work
    this.setState({
      isInternetExplorer: isIE,
      isOldEdge: isOldEdge
    })

  }

  // start the selected experiment
  startExperiment(selectedExp) {

    // get a random condition
    // draw a random condition: 0 = high stress; 1 = low stress
    let condition = Math.floor(Math.random() * 2);

    // get a random task order

    // Helper function that shuffles an array from
    // https://stackoverflow.com/questions/2450954/how-to-randomize-shuffle-a-javascript-array?page=1&tab=votes#tab-top
    const shuffleArray = function(array){
      for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
      }
    };

    let taskOrder = [0, 1, 2, 3, 4];
    shuffleArray(taskOrder);

    // if a valid browser is used to view the experiment
    if (!this.state.isInternetExplorer && !this.state.isEdge) {

      this.setState({
        exp: selectedExp,
        condition: condition,
        taskOrder: taskOrder,
        skipButtonDisabled: false
      })
    } else {
      // if windows edge or internet explorer are used to view the study
      this.setState({
        exp: selectedExp,
        condition: condition,
        taskOrder: taskOrder,
        badBrowser: true,
      })
    }
  }


  // start the experiment if the user uses an old version of the edge browser and wants to continue
  startExpWithEdge() {

    console.log("Experiment started with Edge");

    this.setState({
      badBrowser: false,
      skipButtonDisabled: false,
    })

  }

  // reset the experimental page number to 0 to "soft restart" the online-study
  restartOnlineStudy() {
    this.setState({
      onlineStudy: {restarts: this.state.restarts + 1},
      experimentalPhase: 0
    })
  }


  // go to the next experimental phase (if its not the last page of the experiment)
  nextPage(){

    if (this.state.phaseNumbers - 1 > this.state.experimentalPhase) {
      this.setState({
        experimentalPhase: this.state.experimentalPhase + 1
      })
    }
  }

  // Allow to go to the previous page (if it is not the first page)
  previousPage(){
    if (this.state.experimentalPhase > 0) {
      this.setState({
        experimentalPhase: this.state.experimentalPhase - 1})
    }
  }

  // get the length of the experiment (the number of experimental pages in the experiment)
  getExpLength(len){
    this.setState({phaseNumbers: len})
  }

  // reset the experiment settings
  cancelExperiment() {

    if (this.state.exp) {
      console.log("\n");
      console.log("Study cancelled");
      console.log("\n");

      this.setState({
        exp: false,
        experimentalPhase: 0,
        endOfExperiment: false,
        phaseNumbers: 0,
        restarts: 0,
        skipButtonDisabled: true,
      })
    }

  }

  // disable/enable the next page/previous page button if the media requirements are not met/met
  // (study cant be done on a smartphone/tablet and the screen size must be large enough)
  toggleSkipButton(bool) {
    this.setState({
      skipButtonDisabled: bool
    })
  }

  render() {

    return (
        <div>
          <div className="outerContainer">
            <Navbar cancelExp={() => this.cancelExperiment()}
                    goBackPhase={() => this.previousPage()}
                    proceedPhase={() => this.nextPage()}
                    skipButtonDisabled={this.state.skipButtonDisabled}
            />
            {!this.state.exp ?
                <DemoStartPage startExperiment={this.startExperiment}/>
                :
                this.state.badBrowser ?
                    this.state.isInternetExplorer ?
                        <InternetExplorerWarning/>
                        :
                        <EdgeWarning startExpWithEdge={this.startExpWithEdge}/>
                    :
                    <div className="innerContainer">
                      <div style={{"margin": "auto"}}>
                        {this.state.exp === "OnlineStudy" ?
                            <OnlineExpNavigator
                                phase={this.state.experimentalPhase}
                                condition={this.state.condition}
                                taskOrder={this.state.taskOrder}
                                numRestarts={this.state.onlineStudy.restarts}
                                nextPage={this.nextPage}
                                phaseNumbers={this.getExpLength}
                                restart={()=> this.restartOnlineStudy()}
                                toggleSkipButton={this.toggleSkipButton}
                            />
                            :
                            <LabExpNavigator
                                phase={this.state.experimentalPhase}
                                condition={this.state.condition}
                                taskOrder={this.state.taskOrder}
                                nextPage={this.nextPage}
                                phaseNumbers={this.getExpLength}
                                toggleSkipButton={this.toggleSkipButton}
                            />}
                      </div>
                    </div>
            }
          </div>
        </div>
    );
  }
}
