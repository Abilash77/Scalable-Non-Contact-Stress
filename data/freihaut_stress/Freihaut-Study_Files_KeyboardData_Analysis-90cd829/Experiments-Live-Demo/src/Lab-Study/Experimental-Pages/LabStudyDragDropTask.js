// Drag and Drop Task component


import React, { Component } from 'react';

import KeyboardMouseTracker from '../../General-Components/KeyboardMouseTracker';

import LabStudyDragCircle from '../Layout-Components/LabStudyDragCircle';
import Timer from '../Layout-Components/LabStudyTimer';


const dropTargets = {
    "PR":
        [
            "green",
            "orange",
            "red",
            "blue",
        ],
    "HS": [
        "green",
        "blue",
        "red",
        "blue",
        "orange",
        "green",
        "red",
        "orange",
        "blue",
        "red",
        "orange",
        "blue",
        "green",
        "blue",
        "red",
    ],
    "LS": [
        "green",
        "blue",
        "red",
        "blue",
        "orange",
        "green",
        "red",
        "orange",
        "blue",
        "red",
        "orange",
        "blue",
        "green",
        "blue",
        "red",
    ]
};

export default class LabStudyDragDropTask extends Component {

    constructor(props) {
        super(props);
        this.state = {
            timer: false,
            circlesDragged: 0,
            modal: "modal is-active",
            dragging: false
        };

        this.getDragInfo = this.getDragInfo.bind(this);
        this.endTask = this.endTask.bind(this);
        this.initializeTask = this.initializeTask.bind(this);
        this.returnPositions = this.returnPositions.bind(this);
        this.renderDragCircle = this.renderDragCircle.bind(this);

        this.targets = dropTargets[this.props.phaseName.slice(0,2)];
        this.key = 0;
    }

    componentDidMount() {

        window.scrollTo(0, 0);

        // setup variable to store tracker data
        this.mouseKeyboardData = [];

        if (this.props.phaseName.slice(0,2) !== "PR") {
            this.initializeTask();
        }

        // add is mounted bool to prevent set state attempts after the component is already unmounted
        this._isMounted = true;
    }

    componentWillUnmount() {
        this._isMounted = false;
    }

    initializeTask() {
        this.setState({
            modal: "modal",
            timer: true
        });
    }

    // Check if the Draggable Circle is dropped Inside the Target Circle
    returnPositions(xPos, yPos) {
        let targetX;
        let targetY;
        if(this.targets[this.state.circlesDragged] === "green") {
            targetX = 35;
            targetY = 35;
        } else if (this.targets[this.state.circlesDragged] === "orange") {
            targetX = 1060;
            targetY = 35;
        } else if (this.targets[this.state.circlesDragged] === "red") {
            targetX = 35;
            targetY = 590;
        } else if (this.targets[this.state.circlesDragged] === "blue") {
            targetX = 1060;
            targetY = 590;
        }

        let distToCircleCenter = Math.pow(targetX - xPos, 2) + Math.pow(targetY - yPos, 2);
        if (distToCircleCenter < Math.pow(10, 2)) {
            if (this.state.circlesDragged < this.targets.length - 1) {
                this.key ++;
                this.setState({
                    circlesDragged: this.state.circlesDragged + 1
                })
            } else {
                this.endTask();
            }
        } else {
            // Return the circle to its initial position
            this.key ++;
            this.setState(this.state);
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
            page: this.props.phaseName,
            circlesDragged: this.state.circlesDragged,
            circleNumber: this.key,
            dragging: this.state.dragging
        };

        Object.assign(datapoint, pageInfo);
        this.mouseKeyboardData.push(datapoint);
    }

    renderDragCircle() {
        return(<LabStudyDragCircle x="550" y="315"
                                   dragging={this.getDragInfo}
                                   fill={this.targets[this.state.circlesDragged]}
                                   key={this.key}
                                   returnPositions={this.returnPositions}/>)

    }

    endTask(){
        let data = {mouseKeyboardData: this.mouseKeyboardData}

        this.props.proceedPhase(this.props.phaseName, data);
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
                            <h3>Practice task "Circle Dragging"</h3>
                            <p>
                                In this task, you have to drag and drop a colored circle into the square with the same color.
                            </p>
                            <p>
                                You can drag the circle by clicking on it with the left mouse button and moving the mouse while keeping the mouse button clicked down. Release the mouse button to drop the circle.
                            </p>
                            <p>
                                If you drop the circle outside the same colored square, the position of the circle will reset to the start. The position of the circle will also reset to the start, if you drag the circle outside of the marked playing field.
                            </p>
                            <p>
                                As soon as the circle is dropped in the same colored square, a new circle will appear at the starting position.
                            </p>
                            <p> The task ends after you have successfully dragged and dropped all circles or after the countdown has expired
                            </p>
                        </div>
                    </section>
                    <footer className="modal-card-foot">
                        <button className="button is-link" onClick={this.initializeTask}>Start</button>
                    </footer>
                </div>
            </div>
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
                    <strong>Drag and drop the circle in the same colored square</strong>
                </p>
                <p style={{
                    WebkitUserSelect: "none",
                    MozUserSelect: "none",
                    msUserSelect: "none",
                    UserSelect: "none"
                }}>Remaining time: {!this.state.timer ?
                    this.props.phaseName.slice(0,2) === "PR" ?
                        "25" : "60"
                    : <Timer time={this.props.phaseName.slice(0,2) === "PR" ? 25 : 60} end={()=> this.endTask()}/>}</p>
                <svg style={ {height: "630px",
                    width: "1100px",
                    border: "solid 3px black",
                    marginTop: "10px",
                    cursor: this.state.dragging ? "grabbing" : "grab"}}>
                    <rect x="10" y="10" width="50" height="50" stroke="green" strokeWidth="5px" fill="white"/>
                    <rect x="10" y="565" width="50" height="50" stroke="red" strokeWidth="5px" fill="white"/>
                    <rect x="1035" y="10" width="50" height="50" stroke="orange" strokeWidth="5px" fill="white"/>
                    <rect x="1035" y="565" width="50" height="50" stroke="blue" strokeWidth="5px" fill="white"/>
                    {this.renderDragCircle()}
                </svg>
                {this.props.phaseName.slice(0,2) === "PR" ? this.renderInstructionModal() : null}
            </div>
        );
    }

}