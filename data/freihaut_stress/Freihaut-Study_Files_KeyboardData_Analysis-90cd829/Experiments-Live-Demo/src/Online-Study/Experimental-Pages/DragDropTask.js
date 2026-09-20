import React, { Component } from 'react';

import DragCircle from '../Layout-Components/DragCircle';
import KeyboardMouseTracker from "../../General-Components/KeyboardMouseTracker";
import EndofPracticeModal from "../Layout-Components/EndofPracticeModal";


export default class DragDropTask extends Component {

    constructor(props) {
        super(props);

        // initial targets are the practice targets
        const practiceTrials = [
            [35, 35], [860, 35], [860, 490], [35, 490]
        ];

        this.state = {
            circlesDragged: 0,
            targets: practiceTrials,
            modal: "modal is-active",
            dragging: false,
            startModal: "modal",
            practice: true,
            onTarget: false
        };

        this.getDragInfo = this.getDragInfo.bind(this);
        this.endTask = this.endTask.bind(this);
        this.initializeTask = this.initializeTask.bind(this);
        this.returnPositions = this.returnPositions.bind(this);
        this.renderDragCircle = this.renderDragCircle.bind(this);
        this.initializePractice = this.initializePractice.bind(this);
        this.onTarget = this.onTarget.bind(this);

        this.key = 0;
    }

    componentDidMount() {

        window.scrollTo(0, 0);

        document.body.classList.add("is-clipped");

        // setup variable to store tracker data
        this.mouseKeyboardData = [];
        // get the start time of the component
        this.startTime = Date.now();

        const phase = this.props.name.slice(0, 2);

        if (phase === "Co") {
            this.initializeTask();
        }

        // add is mounted bool to prevent set state attempts after the component is already unmounted
        this._isMounted = true;
    }

    componentWillUnmount() {

        this._isMounted = false;
    }

    initializePractice() {

        document.body.classList.remove("is-clipped");

        this.setState({
            modal: "modal",
        });
    }

    initializeTask() {

        this.trialStartTime = Date.now();
        this.key = 0;

        const trialTargets =  [
            [35, 490],
            [860, 490],
            [35, 35],
            [860, 35],
            [860, 490],
            [860, 35],
            [35, 490],
            [860, 35],
            [860, 490],
            [35, 35],
            [35, 490],
            [860, 35],
        ];

        document.body.classList.remove("is-clipped");

        this.setState({
            modal: "modal",
            startModal: "modal",
            practice: false,
            circlesDragged: 0,
            targets: trialTargets
        });
    }

    restart() {
        document.body.classList.add("is-clipped");
        this.setState({
            modal: "modal is-active",
            startModal: "modal",
            circlesDragged: 0,
            onTarget: false
        })

    }

    // Check if the Draggable Circle is dropped Inside the Target Circle
    returnPositions(xPos, yPos) {
        let targetX = this.state.targets[this.state.circlesDragged][0];
        let targetY = this.state.targets[this.state.circlesDragged][1];

        let distToCircleCenter = Math.pow(targetX - xPos, 2) + Math.pow(targetY - yPos, 2);
        if (distToCircleCenter < Math.pow(10, 2)) {
            if (this.state.circlesDragged < this.state.targets.length - 1) {
                this.key ++;
                this.setState({
                    circlesDragged: this.state.circlesDragged + 1,
                    onTarget: false
                })
            } else {
                if (this.state.practice) {
                    this.key ++;
                    document.body.classList.add("is-clipped");
                    this.setState({
                        startModal: "modal is-active",
                        onTarget: false
                    })
                } else {
                    this.endTask();
                }
            }
        } else {
            // Return the circle to its initial position
            this.key ++;
            this.setState(this.state);
        }
    }

    onTarget(x, y){

        let targetX = this.state.targets[this.state.circlesDragged][0];
        let targetY = this.state.targets[this.state.circlesDragged][1];

        let distToCircleCenter = Math.pow(targetX - x, 2) + Math.pow(targetY - y, 2);
        if (distToCircleCenter < Math.pow(10, 2)) {
            this.setState({
                onTarget: true
            })
        } else {
            this.setState({
                onTarget: false
            })
        }

    }

    getDragInfo(bool) {
        if (this._isMounted) {
            if (bool){
                this.setState({dragging: true})
            } else {
                this.setState({dragging: false})
            }
        }
    }

    onKeyboardMouseEvent(datapoint) {
        // save mouse data and add page relevant info before pushing it into the data array
        // Info about the trial number and if the circle is being dragged or not
        const pageInfo = {
            circlesDragged: this.state.circlesDragged,
            circleNumber: this.key,
            dragging: this.state.dragging,
            practice: this.state.practice,
            onTarget: this.state.onTarget
        };

        Object.assign(datapoint, pageInfo);
        // console.log(datapoint);
        this.mouseKeyboardData.push(datapoint);
    }

    renderDragCircle() {
        return(<DragCircle x="450" y="265"
                           dragging={this.getDragInfo}
                           key={this.key}
                           returnPositions={this.returnPositions}
                           onTarget={this.onTarget}
        />)

    }

    endTask(){

        const data = {"taskLoaded": this.startTime, "trialStarted": this.trialStartTime, "taskEnded": Date.now(),
            "TrackerData": this.mouseKeyboardData};

        this.props.proceedPhase(this.props.name, data);
    }

    renderInstructionModal(){
        return (
            <div className={this.state.modal} style={{textAlign: "left", fontSize: "18px"}}>
                <div className="modal-background">{null}</div>
                <div className="modal-content">
                    <header className="modal-card-head">
                        <p className="modal-card-title">{null}</p>
                    </header>
                    <section className="modal-card-body">
                        <div className="content">
                            <h3>Task "Circle Dragging"</h3><br/>
                            <p>
                                In this task you have to <strong>drag and drop a circle into a squared target field</strong>.
                            </p>
                            <p>
                                You drag the circle by clicking on it with the left mouse button and then moving your mouse while holding down the left mouse button. To drop the circle, release the left mouse button.
                            </p>
                            <p>
                                The squared target field changes <span style={{color:"LightBlue", fontWeight: "bold"}}>color</span>, as soon as you can drop the circle there. If you accidentally drop the circle outside of the target field, it returns to its starting position. The circle also return to its starting position if you drag it outside of the marked playing field.
                            </p>
                            <p>
                                The task ends after a few trials. The remaining number of trials will be displayed
                            </p>
                        </div>
                    </section>
                    <footer className="modal-card-foot">
                        <button className="button is-link" onClick={this.initializePractice}>Start practice task</button>
                    </footer>
                </div>
            </div>
        )
    }

    renderStartModal(){
        return(
            <EndofPracticeModal startModal={this.state.startModal} restart={() => this.restart()} initializeTask={this.initializeTask}/>
        )
    }


    render() {
        return(
            <div>
                <KeyboardMouseTracker onEvent={this.onKeyboardMouseEvent.bind(this)}/>
                <p style={{
                    WebkitUserSelect: "none",
                    MozUserSelect: "none",
                    msUserSelect: "none",
                    UserSelect: "none"
                }}>
                    <strong>Drag the circle into the square</strong>
                </p>
                <p style={{
                    WebkitUserSelect: "none",
                    MozUserSelect: "none",
                    msUserSelect: "none",
                    UserSelect: "none"
                }}>Remaining trials: {this.state.targets.length - this.state.circlesDragged} of {this.state.targets.length}</p>
                <svg style={ {height: "530px",
                    width: "900px",
                    border: "solid 3px black",
                    marginTop: "10px",
                    cursor: this.state.dragging ? "grabbing" : "grab",
                    WebkitUserSelect: "none",
                    MozUserSelect: "none",
                    msUserSelect: "none",
                    UserSelect: "none"}}>
                    <rect x={this.state.targets[this.state.circlesDragged][0]-25}
                          y={this.state.targets[this.state.circlesDragged][1]-25}
                          width="50" height="50" stroke="black" strokeWidth="5px" fill={this.state.onTarget ? "LightBlue" : "white"}
                    style={{ WebkitUserSelect: "none",
                        MozUserSelect: "none",
                        msUserSelect: "none",
                        UserSelect: "none"}}/>
                    {this.renderDragCircle()}
                </svg>
                {this.renderInstructionModal()}
                {this.renderStartModal()}
            </div>
        );
    }

}