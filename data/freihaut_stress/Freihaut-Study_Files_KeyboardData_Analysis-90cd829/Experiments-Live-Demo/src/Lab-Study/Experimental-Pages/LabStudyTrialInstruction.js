// start page of the trials

import React, {Component} from 'react';


import LabStudyProceedButton from '../Layout-Components/LabStudyProceedButton';

export default class LabStudyTrialInstruction extends Component {

    constructor(props) {
        super(props);
        this.state = {
            checked: false
        };

        this.toggleCheckBox = this.toggleCheckBox.bind(this);
    }

    componentDidMount() {
        window.scrollTo(0, 0);
    }

    toggleCheckBox() {
        this.state.checked ? this.setState({checked: false}) : this.setState({checked: true})
    }

    renderInstruction() {
      // Return the Correct Instruction Text
        if (this.props.phaseName.slice(0,2) === "PR") {
            return(
                <div className="content">
                    <h3>Practice Block</h3><br/>
                    <h4>In the following, you will practice each task.</h4>
                    <h4>Before the start of a task, you will receive detailed instructions about how to do the task. Please read the instructions carefully.</h4>
                    <h4>If you have questions regarding a task, please ask the experimenter. Please be sure that you know how to do each task, because you cannot ask the experimenter after the practice anymore.</h4><br/>
                    <h4>If you are ready to start the practice, click on „continue“.</h4>
                    <LabStudyProceedButton onClick={() => this.props.proceedPhase(this.props.phaseName)}/>
                </div>
            )
        } else if (this.props.phaseName.slice(0,2) === "LS") {
            return(
                <div className="content">
                    <h3>Exercise Block {this.props.order === 0 ? 2 : 1}</h3><br/><br/>
                    <h4>The following exercise block <span style={{textDecoration: "underline", fontWeight: "bolder"}}>is no performance test.</span></h4><br/>
                    <h4>As soon as you are ready to start the exercise block, click on „continue“.</h4>
                    <LabStudyProceedButton onClick={() => this.props.proceedPhase(this.props.phaseName)}/>
                </div>
            )
        } else if (this.props.phaseName.slice(0,2) === "HS") {
            return (
                <div className="content">
                    <h3>Exercise Block {this.props.order === 0 ? 1 : 2}</h3><br/>
                    <h4>The following exercise block <span style={{textDecoration: "underline", fontWeight: "bolder"}}>is a performance test.</span></h4>
                    <h4><span className="has-text-danger">Note:</span> Your performance will be continuously measured, evaluated and compared with a target value during this exercise block. If your performance is below the target value, the experimenter will perform an additional performance test after this exercise block.</h4>
                    <h5 className="checkbox" style={{marginTop: "25px"}}>
                        <input type="checkbox" style={{marginRight: "5px"}} checked={this.state.checked} onChange={this.toggleCheckBox}/>
                        I agree to the performance test and I am ready to start
                    </h5>
                    <LabStudyProceedButton onClick={() => this.props.proceedPhase(this.props.phaseName)} disabled={!this.state.checked}/>
                </div>
            )

        }

    }

    render() {
        return (
            <div className="section">
                {this.renderInstruction(this.props.phaseName.slice(0,2))}
            </div>
        );
    }

}