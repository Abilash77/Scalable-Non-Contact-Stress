import React, { Component } from 'react';

// import the Online-Study Components
import IntroPage from "./Experimental-Pages/IntroPage";
import TrialInstruction from './Experimental-Pages/TrialInstruction';
import TrialStart from "./Experimental-Pages/TrialStart";
import Debriefing from "./Experimental-Pages/Debriefing";
import DonationOption from "./Experimental-Pages/DonationOption";
import End from "./Experimental-Pages/End";
// Questions
import Soziodem from './Experimental-Pages/Soziodem';
import Mdbf from './Experimental-Pages/Mdbf';
import SamStress from './Experimental-Pages/Sam';
// Keyboard & Mouse Tasks
import DragDropTask from './Experimental-Pages/DragDropTask';
import FollowBoxTask from './Experimental-Pages/FollowBoxTask';
import PatternTyping from './Experimental-Pages/PatternTyping';
import Slider from './Experimental-Pages/Slider_Captcha';
import PointClickTask from './Experimental-Pages/PointClickTask';
// Stress manipulation
import CountTask from "./Experimental-Pages/CountTask";
import CountTaskAnswer from "./Experimental-Pages/CountTaskAnswer";

// import the component that is shown when the screen size is too small or a smartphone or tablet is used
// to for the study
import MediaMismatch from "../General-Components/MediaMismatch";

export default class OnlineExpNavigator extends Component {

    constructor(props) {
        super(props);

        this.state = {
            lastPage: false,
            pageTime: Date.now(),
            countTaskAnswer: 0,
            countTaskSolution: 0,
            hideCursor: false,
            submitExpLength: false,
            failsMediaCheck: false
        };

        this.proceedPhase = this.proceedPhase.bind(this);
        this.updateScore = this.updateScore.bind(this);
        this.updateAns = this.updateAns.bind(this);
        this.getExpPages = this.getExpPages.bind(this);
    }

    componentDidMount() {

        // get the length of the experiment (number of experimental pages)
        const expLength = this.getExpPages().length;
        // save the info
        this.props.phaseNumbers(expLength);

        // get the start time of the experiment
        this.expStartTime = Date.now();

        // Get and Log some meta-data for the start of the experiment
        // check if the browser is internet explorer or edge
        // Internet Explorer 6-11
        const isIE = /*@cc_on!@*/false || !!document.documentMode;
        // Edge 20+
        const isOldEdge = !isIE && !!window.StyleMedia;

        const experimentMetaData = {
            ExpStartTime: this.expStartTime,
            userAgent: navigator.userAgent,
            condition: this.props.condition,
            taskOrder: this.props.taskOrder,
            initScreenProps: {height: window.innerHeight, width: window.innerWidth},
            initZoom: window.devicePixelRatio * 100,
            isInternetExplorer: isIE,
            isOldEdge: isOldEdge
        };

        // Log the experiment metadata for debugging
        console.log("Start of the Online-Study.");
        console.log("Initial Experiment Meta Data:");
        console.log(experimentMetaData);

        // use the css media query programmatically (check whether the screen size is smaller than allowed and whether
        // there is a pointer device and hover): if the test fails, don't allow the participant to continue and show
        // a message that the study requirements are not matched
        let mql = window.matchMedia('only screen and (max-width: 1024px), screen and (max-height: 600px), not (pointer: fine), not (hover: hover)');
        // initially check the media query
        if (mql.matches) {
            this.setState({
                failsMediaCheck: true
            }, () => this.props.toggleSkipButton(true))
        } else {
            this.setState({
                failsMediaCheck: false
            }, () => this.props.toggleSkipButton(false))
        }

        // add the event listener to trigger the mismatch event whenever the screen size is adjusted and doesnt match
        // the requirements any more
        mql.addListener(ev => {
            if (ev.matches) {
                this.setState({
                    failsMediaCheck: true
                }, () => this.props.toggleSkipButton(true))
            } else {
                this.setState({
                    failsMediaCheck: false
                }, () => this.props.toggleSkipButton(false))
            }
        });

    }

    proceedPhase(phaseName, data) {

        // Get Meta-Data about the page
        const pageEndTime = Date.now();
        // in seconds
        const pageDuration = (pageEndTime - this.state.pageTime) / 1000;
        const screenProps = {height: window.innerHeight, width: window.innerWidth};
        const pageZoom = window.devicePixelRatio * 100;

        // log the page Metadata
        console.log("\n");
        console.log("LabStudyEnd of " + phaseName);
        console.log("Timestamp", pageEndTime);
        console.log("Page Duration: ", pageDuration);
        console.log("LabStudyEnd of Page Screen Props: ", screenProps);
        console.log("LabStudyEnd of Page Zoom: ", pageZoom);

        // save page data (if there is data on the page)
        if (data) {

            // Log the Data on the Page
            console.log("Page Data: ", data);
        }

        // if the last page is reached remove the unbeforeunload event listener
        if (phaseName === "Debriefing") {
            // log the study duration
            console.log("\n");
            console.log("Final Page reached");
            console.log("Study duration in sec:", (Date.now() - this.expStartTime) / 1000);
            console.log("Number of study restarts:", this.props.numRestarts);
        }

        // go to the next page (add info if its the last page to disable the proceed page button on the last page)
        this.setState({
            pageTime: pageEndTime,
            lastPage: phaseName === "Debriefing"
        }, () => this.props.nextPage());

    }

    // Update score of count task after the task
    updateScore(taskNum) {
        this.setState({
            countTaskSolution: this.state.countTaskSolution + taskNum
        })
    }

    updateAns(ans) {
        this.setState({
            countTaskAnswer: this.state.countTaskAnswer + ans
        })
    }

    // hide the mouse cursor during the count task
    hideMouseCursor() {
        this.setState({
            hideCursor: true
        })
    }

    // show the mouse cursor again after the count task
    showMouseCursor() {
        this.setState({
            hideCursor: false
        })
    }

    // if the participant has no mouse and wants to restart the study with a mouse
    // goes back to the first study page and a restart counter is incremented
    restart() {
        console.log("\n");
        console.log("Participant restarts the study");
        console.log("Number of restarts:", this.props.numRestarts + 1);
        this.props.restart()
    }

    getExpPages() {
        // get all experimental pages and render the recent one
        // if the last page has not been reached yet, alert participant that their data is lost on page unload, else
        // if the last page was reached, turn off the alert
        if (this.state.lastPage) {
            window.onbeforeunload = null;
        } else {
            // add a warning message to unloading that data is not saved
            window.onbeforeunload = function () {
                return "Sind Sie sicher, dass Sie diese Seite verlassen möchten? Das Verlassen der Seite beendet die" +
                    "Studie automatisch. Sie können die Studie nicht an dieser Stelle fortsetzen und Ihre Daten gehen verloren.";
            };
        }

        // Helper function that adds new items (of an array) to another array starting at a specific index
        const insert = (arr, index, newItems) => [
            // part of the array before the specified index
            ...arr.slice(0, index),
            // inserted items
            ...newItems,
            // part of the array after the specified index
            ...arr.slice(index)
        ];

        //store all Mouse and Keyboard Tasks
        // if changes to this array are made, the task order array in the constructor needs to be adapted, too!
        let mouseKeyboardTasks = [
            {name: "PointClick", page: <PointClickTask proceedPhase={this.proceedPhase}/>},
            {name: "FollowBox", page:  <FollowBoxTask proceedPhase={this.proceedPhase} />},
            {name: "DragDrop", page: <DragDropTask proceedPhase={this.proceedPhase}/>},
            {name: "Slider", page: <Slider proceedPhase={this.proceedPhase}/>},
            {name: "PatternTyping", page: <PatternTyping proceedPhase={this.proceedPhase}/>}
        ];

        // reorder the task array according to the randomized order created in the constructor
        mouseKeyboardTasks =  this.props.taskOrder.map(i => mouseKeyboardTasks[i]);

        // Store all Condition pages
        const conditionTrial = [
            <TrialInstruction name={"Con_Instr"} condition={this.props.condition} proceedPhase={this.proceedPhase}/>,
            <CountTask name={"Pr_Count"} key={0} order={0} updateScore={this.updateScore} proceedPhase={this.proceedPhase}
                       condition={this.props.condition} showMouseCursor={() => this.showMouseCursor()} hideMouseCursor={() => this.hideMouseCursor()}/>,
            <TrialStart name={"TrialStart"} condition={this.props.condition} proceedPhase={this.proceedPhase}/>
        ];

        // store all paged of the Practice Trial
        const practiceTrial = [
            <TrialInstruction name={"Pr_Instruction"} condition={this.props.condition} proceedPhase={this.proceedPhase}/>,
        ];

        // fill the high and low stress condition arrays with trios of a Mental Arithmetic Task, a Mouse or Keyboard Task
        // and a SAM
        for (let i=0; i < mouseKeyboardTasks.length; i++) {
            // get the name of the phase
            let taskName = mouseKeyboardTasks[i].name;
            let task = mouseKeyboardTasks[i].page;

            // add tasks together with stressor and sam to the trial pages
            conditionTrial.push(
                <CountTask name={"Con_Count_" + taskName} key={i+1} order={i} updateScore={this.updateScore} proceedPhase={this.proceedPhase}
                           condition={this.props.condition} showMouseCursor={() => this.showMouseCursor()} hideMouseCursor={() => this.hideMouseCursor()}/>,
                <CountTaskAnswer name={"Con_CountAns_" + taskName} proceedPhase={this.proceedPhase} updateAns={this.updateAns}/>,
                React.cloneElement(task, {name: "Con_" + taskName}),
                <SamStress name={"Con_Sam_" + taskName} proceedPhase={this.proceedPhase}/>
            );

            // add tasks together with sam to the practice pages
            practiceTrial.push(
                React.cloneElement(task, {name: "Pr_" + taskName}),
                <SamStress name={"Pr_Sam_" + taskName} proceedPhase={this.proceedPhase}/>
            );
        }

        // Add the end pages to the LabStudyEnd of the trial pages
        conditionTrial.push(
            <Mdbf name={"Con_Mdbf"} proceedPhase={this.proceedPhase}/>,
        );

        // add the end pages to the end of the practice pages
        practiceTrial.push(
            <Mdbf name={"Pr_Mdbf"} proceedPhase={this.proceedPhase}/>,

        );

        // Create arrays of "standard pages (that are not shuffled and dont depend on the condition)
        let startPages = [
            <IntroPage name={"Intro"} proceedPhase={this.proceedPhase}/>,
            <Soziodem name={"Soziodem"} proceedPhase={this.proceedPhase} restart={()=> this.restart()}/>,
        ];

        let endPages = [
            <DonationOption name={"DonationOption"} proceedPhase={this.proceedPhase}/>,
            <Debriefing name={"Debriefing"} condition={this.props.condition} proceedPhase={this.proceedPhase}
                        ans={this.state.countTaskAnswer} solution={this.state.countTaskSolution}/>,
            <End name={"EndPage"}/>,
        ];

        // initialize the Experimental Flow (order of all pages)
        let expFlow;

        // add all arrays with experimental pages to the expFlow starting chronologically from the beginning of the experiment (order in array is order
        // of shown pages
        expFlow = startPages;

        // add the practice pages to the expFlow
        expFlow = insert(expFlow, expFlow.length, practiceTrial);
        // add the conditian trial pages
        expFlow = insert(expFlow, expFlow.length, conditionTrial);
        // add the end pages
        expFlow = insert(expFlow, expFlow.length, endPages);

        return expFlow

    }

    render() {

        let pages = this.getExpPages();

        return (
            <div>
                {this.state.failsMediaCheck
                    ?
                    <MediaMismatch/>
                    :
                    <div style={this.state.hideCursor ? {cursor: "none"} : {}}>
                        {pages[this.props.phase]}
                    </div>
                }
            </div>
        )
    }

}