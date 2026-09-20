import React, {Component} from 'react';

export default class TrialStart extends Component {


    componentDidMount() {
        window.scrollTo(0, 0);
    }



    render() {
        return (
            <div className="section">
                <div className="content">
                    <h3>
                        Start of the {this.props.condition === 0 ? "test" : "application"}
                    </h3>
                    <h4>
                        The {this.props.condition === 0 ? "test" : "application"} will start now. In the test, the “Counting Squares” task is added to each of the previously practiced tasks (e.g. "Circle Clicking", "Follow-the-Circle", "Typing").</h4>
                    <h4>
                        After each “Counting Squares” task, you will be asked to indicate how many squares you have counted.
                    </h4>
                    <div style={{
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        marginTop: "58px",
                    }} className="control">
                        <button className="button is-link" onClick={() => this.props.proceedPhase(this.props.name)}>
                            Start the {this.props.condition === 0 ? "test" : "application"}
                        </button>
                    </div>
                </div>
            </div>
        );
    }

}