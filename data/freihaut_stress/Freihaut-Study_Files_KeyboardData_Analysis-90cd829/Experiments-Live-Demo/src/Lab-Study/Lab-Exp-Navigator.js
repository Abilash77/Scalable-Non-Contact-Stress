// React Imports
import React, { Component } from 'react';

// import the lab-study components (experiment pages)
import LabStudyIntroPage from './Experimental-Pages/LabStudyIntroPage';
import LabStudySoziodem from './Experimental-Pages/LabStudySoziodem';
import LabStudyEnd from "./Experimental-Pages/LabStudyEnd";
import LabStudyBfiNeuroticism from './Experimental-Pages/LabStudyBfiNeuroticism';
import SamStress from './Experimental-Pages/LabStudySam';
import LabStudyDrawingTask from './Experimental-Pages/LabStudyDrawingTask';
import LabStudyTrialInstruction from './Experimental-Pages/LabStudyTrialInstruction';
import LabStudyTrialEnd from './Experimental-Pages/LabStudyTrialEnd';
import LabStudyMdbf from './Experimental-Pages/LabStudyMdbf';
import LabStudyQuestionsAfterTasks from './Experimental-Pages/LabStudyQuestionsAfterTasks';
// Tasks
import LabStudyDragDropTask from './Experimental-Pages/LabStudyDragDropTask';
import LabStudyFollowBoxTask from './Experimental-Pages/LabStudyFollowBoxTask';
import LabStudyPatternTyping from './Experimental-Pages/LabStudyPatternTyping';
import LabStudyPointClickTask from './Experimental-Pages/LabStudyPointClickTask';
import LoadingTask from './Experimental-Pages/LabStudyLoadingTask';
//Stress Manipulation Task
import MentalArithmetic from './Experimental-Pages/LabStudyMentalArithmetics';
import MediaMismatch from "../General-Components/MediaMismatch";


export default class OnlineExpNavigator extends Component {

    constructor(props) {
        super(props);

        this.state = {
            lastPage: false,
            pageTime: Date.now(),
            scorePractice: 0,
            scoreHighStress: 0,
            scoreLowStress: 0,
            taskNumLowStress: 0,
            taskNumHighStress: 0,
            submitExpLength: false,
            failsMediaCheck: false
        };

        this.proceedPhase = this.proceedPhase.bind(this);
        this.updateScore = this.updateScore.bind(this);
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

        // Log the experiment metadata
        console.log("Start of the Lab-Study.");
        console.log("Initial Experiment Meta Data:");
        console.log(experimentMetaData);

        // use the css media query programmatically (check whether the screen size is smaller than allowed and whether
        // there is a pointer device and hover): if the test fails, don't allow the participant to continue and show
        // a message that the study requirements are not matched
        let mql = window.matchMedia('only screen and (max-width: 1150px), screen and (max-height: 700px), not (pointer: fine), not (hover: hover)');
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
        if (phaseName === "LabStudyBfiNeuroticism") {
            // log the study duration
            console.log("\n");
            console.log("Final Page reached");
            console.log("Study duration in sec:", (Date.now() - this.expStartTime) / 1000);
        }

        // go to the next page (add info if its the last page to disable the proceed page button on the last page)
        this.setState({
            pageTime: pageEndTime,
            lastPage: phaseName === "LabStudyBfiNeuroticism"
        }, () => this.props.nextPage());

    }

    // handle the mental arithmetic score throughout the experiment
    updateScore(newScore, condition, taskNum) {
        if (condition === "LS") {
            this.setState({
                scoreLowStress: newScore,
                taskNumLowStress: taskNum
            });
        } else if (condition === "HS") {
            this.setState({
                scoreHighStress: newScore,
                taskNumHighStress: taskNum
            });
        }
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
        let mouseKeyboardTasks = [
            {name: "PointClick", page: <LabStudyPointClickTask proceedPhase={this.proceedPhase}/>},
            {name: "Drawing", page: <LabStudyDrawingTask proceedPhase={this.proceedPhase}/>},
            {name: "DragDrop", page: <LabStudyDragDropTask proceedPhase={this.proceedPhase}/>},
            {name: "FollowBox", page: <LabStudyFollowBoxTask proceedPhase={this.proceedPhase}/>},
            {name: "LabStudyPatternTyping", page: <LabStudyPatternTyping proceedPhase={this.proceedPhase}/>},
        ];

        // reorder the task array according to the randomized order created in the constructor
        mouseKeyboardTasks =  this.props.taskOrder.map(i => mouseKeyboardTasks[i]);

        // Store all pages of the High Stress Condition
        let highStressCondition = [
            <LabStudyTrialInstruction phaseName={"HS_TrialInstruction"} order={this.props.condition} proceedPhase={this.proceedPhase}/>,
            <LoadingTask phaseName={"HS_loading_task"} proceedPhase={this.proceedPhase}/>
        ];

        // Store all pages of the Low Stress Condition
        let lowStressCondition = [
            <LabStudyTrialInstruction phaseName={"LS_TrialInstruction"} order={this.props.condition} proceedPhase={this.proceedPhase}/>,
            <LoadingTask phaseName={"LS_loading_task"} proceedPhase={this.proceedPhase}/>,
        ];

        // store all paged of the Practice Trial
        let practiceTrial = [
            <LabStudyTrialInstruction phaseName={"PR_TrialInstruction"} proceedPhase={this.proceedPhase}/>,
            <MentalArithmetic phaseName={"PR_Mental_Arithmetic"} taskNumber={0} score={0} updateScore={this.updateScore} proceedPhase={this.proceedPhase}/>,
        ];

        // fill the high and low stress condition arrays with trios of a Mental Arithmetic Task, a Mouse or Keyboard Task
        // and a LabStudyBfiNeuroticism Task
        for (let i=0; i < mouseKeyboardTasks.length; i++) {
            // get the name of the task
            let taskName = mouseKeyboardTasks[i].name;
            // High Stress Condition
            // push an object trio for each task inside the task array
            highStressCondition.push(
                <MentalArithmetic phaseName={"HS_Mental_Arithmetic_" + taskName} taskNumber={this.state.taskNumHighStress} score={this.state.scoreHighStress} updateScore={this.updateScore} proceedPhase={this.proceedPhase}/>,
                React.cloneElement(mouseKeyboardTasks[i].page, {phaseName: "HS_" + taskName}),
                <SamStress phaseName={"HS_SAM_" + taskName} proceedPhase={this.proceedPhase}/>
            );
            // do same thing for other condition
            lowStressCondition.push(
                <MentalArithmetic phaseName={"LS_Mental_Arithmetic_" + taskName} taskNumber={this.state.taskNumLowStress} score={this.state.scoreLowStress} updateScore={this.updateScore} proceedPhase={this.proceedPhase}/>,
                React.cloneElement(mouseKeyboardTasks[i].page, {phaseName: "LS_" + taskName}),
                <SamStress phaseName={"LS_SAM_" + taskName} proceedPhase={this.proceedPhase}/>
            );
            practiceTrial.push(
                React.cloneElement(mouseKeyboardTasks[i].page, {phaseName: "PR_" + taskName})
            );
        }

        // Add the Trial EndPages to each experimental phase
        highStressCondition.push(
            <LabStudyMdbf phaseName={"HS_MDBF"} proceedPhase={this.proceedPhase}/>,
            <LabStudyTrialEnd phaseName={"HS_TrialEnd"} proceedPhase={this.proceedPhase}/>
        );
        lowStressCondition.push(
            <LabStudyMdbf phaseName={"LS_MDBF"} proceedPhase={this.proceedPhase}/>,
            <LabStudyTrialEnd phaseName={"LS_TrialEnd"} proceedPhase={this.proceedPhase}/>
        );
        practiceTrial.push(
            <LabStudyTrialEnd phaseName={"PR_TrialEnd"} proceedPhase={this.proceedPhase}/>,
        );

        // Store all default pages (that are not shuffeled and dont depend on the condition
        let startPages = [
            <LabStudyIntroPage phaseName={"LabStudyIntroPage"} proceedPhase={this.proceedPhase}/>,
            <LabStudySoziodem phaseName={"LabStudySoziodem"} proceedPhase={this.proceedPhase}/>,
        ];
        let endPages = [
            <LabStudyQuestionsAfterTasks phaseName={"LabStudyQuestionsAfterTasks"} proceedPhase={this.proceedPhase}/>,
            <LabStudyBfiNeuroticism phaseName={"LabStudyBfiNeuroticism"} proceedPhase={this.proceedPhase}/>,
            <LabStudyEnd/>
            ];

        // initialize the Experimental Flow (order of all pages)
        let expFlow;

        // add the start pages to the expFlow
        expFlow = startPages;

        // add the practice pages to the expFlow
        expFlow = insert(expFlow, startPages.length, practiceTrial);

        // add the high- and low-stress trials depending on the condition
        // If the condition is 0 = stress first
        if (this.props.condition === 0) {
            // add the high stress and then the low stress condition pages to the default pages
            expFlow = insert(expFlow, expFlow.length, highStressCondition);
            expFlow = insert(expFlow, expFlow.length + highStressCondition.length, lowStressCondition);
            // if the condition is 1 = low stress
        } else if (this.props.condition === 1) {
            // add the low stress and then high stress condition pages to the default pages
            expFlow = insert(expFlow, expFlow.length, lowStressCondition);
            expFlow = insert(expFlow, expFlow.length + lowStressCondition.length, highStressCondition);
        }

        // add the end pages to the experimental Flow
        expFlow = insert(expFlow, expFlow.length, endPages);

        // return all experimental pages
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
                    <div>
                        {pages[this.props.phase]}
                    </div>
                }
            </div>
        )
    }


}